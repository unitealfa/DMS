# NOTE:
# Cette cellule/script suppose que l'étape précédente a déjà produit:
# - TEXT_FILES (fichiers "text")
# - DOCS (images prétraitées, avec "prepped")
#
# Ici on fait:
# 1) OCR Tesseract UNIQUEMENT sur DOCS (images)
# 2) Extraction NATIVE (sans OCR) sur TEXT_FILES
#
# Objectif print:
# - 1 seule fois à la fin
# - affiche: fichier, nb pages, puis texte brut de chaque page
# - préserver espaces, tabs, lignes vides au maximum

import uuid
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytesseract
from pytesseract import Output


# ==================== Réglage PRINT ====================
# False => aucune sortie pendant OCR/native
# True  => debug pendant extraction
PRINT_DURING_EXTRACTION = False


# ==================== Fallbacks (prod-safe) ====================
# Quand exécuté via orchestrateur, ces globals sont déjà fournis.
# Ici on met des defaults uniquement si manquants.
if "config_flags" not in globals():
    config_flags = []

if "args" not in globals():
    args = SimpleNamespace(
        oem=None,
        psm=None,
        dpi=None,
        tessdata_dir=None,
        user_words=None,
        user_patterns=None,
        lang="fra",
    )

if "build_config" not in globals():
    def build_config(oem, psm, base_flags, dpi, tessdata_dir, user_words, user_patterns):
        # Fallback minimal: renvoie une config vide
        return ""

if "detect_path_type" not in globals():
    class _DummyFT:
        def __init__(self, ext: str):
            self.ext = ext

    def detect_path_type(path: str):
        return _DummyFT(Path(path).suffix.lower())


# ==================== Tesseract flags (préservation layout) ====================
if "-c preserve_interword_spaces=1" not in config_flags:
    config_flags.append("-c preserve_interword_spaces=1")
if "-c textord_tabfind_find_tables=1" not in config_flags:
    config_flags.append("-c textord_tabfind_find_tables=1")


# ==================== OCR layout reconstruction ====================
def _median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _estimate_char_metrics(data: dict) -> Tuple[float, float]:
    widths = []
    heights = []
    texts = data.get("text", [])
    confs = data.get("conf", [])
    ws = data.get("width", [])
    hs = data.get("height", [])

    for i, t in enumerate(texts):
        if t is None:
            continue
        s = str(t)
        # On ignore les tokens vides; ça n'enlève pas des espaces "réels" du rendu final,
        # car les espaces sont reconstruits depuis les positions x/y.
        if not s.strip():
            continue

        try:
            c = float(confs[i])
        except Exception:
            c = 0.0
        if c < 0:
            continue

        w = int(ws[i]) if i < len(ws) else 0
        h = int(hs[i]) if i < len(hs) else 0
        if h > 0:
            heights.append(h)

        L = len(s)
        if w > 0 and L > 0:
            widths.append(w / float(L))

    char_w = _median(widths) or 10.0
    line_h = _median(heights) or 20.0

    if char_w <= 1:
        char_w = 10.0
    if line_h <= 1:
        line_h = 20.0

    return float(char_w), float(line_h)


def _render_layout_from_data(data: dict, img_w: int, img_h: int) -> str:
    char_w, line_h = _estimate_char_metrics(data)
    line_tol = max(6.0, line_h * 0.55)

    items = []
    texts = data.get("text", [])
    confs = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    for i, t in enumerate(texts):
        if t is None:
            continue
        s = str(t)
        if not s.strip():
            continue

        try:
            c = float(confs[i])
        except Exception:
            c = 0.0
        if c < 0:
            continue

        l = int(lefts[i]) if i < len(lefts) else 0
        tp = int(tops[i]) if i < len(tops) else 0
        w = int(widths[i]) if i < len(widths) else 0
        h = int(heights[i]) if i < len(heights) else 0

        items.append({"text": s, "left": l, "top": tp, "right": l + w, "height": h})

    items.sort(key=lambda x: (x["top"], x["left"]))

    # Group into lines
    lines = []
    for it in items:
        placed = False
        if lines and abs(it["top"] - lines[-1]["top"]) <= line_tol:
            lines[-1]["words"].append(it)
            lines[-1]["top"] = min(lines[-1]["top"], it["top"])
            placed = True
        if not placed:
            for ln in reversed(lines):
                if abs(it["top"] - ln["top"]) <= line_tol:
                    ln["words"].append(it)
                    ln["top"] = min(ln["top"], it["top"])
                    placed = True
                    break
        if not placed:
            lines.append({"top": it["top"], "words": [it]})

    lines.sort(key=lambda ln: ln["top"])

    out_lines = []
    prev_row = None

    for ln in lines:
        words = sorted(ln["words"], key=lambda x: x["left"])
        row = int(round(ln["top"] / line_h)) if line_h > 0 else 0

        # Insert blank lines if vertical gap
        if prev_row is not None:
            gap = row - prev_row
            if gap > 1:
                for _ in range(gap - 1):
                    out_lines.append("")
        prev_row = row

        line_str = ""
        cursor = 0
        for w in words:
            col = int(round(w["left"] / char_w)) if char_w > 0 else 0
            if col < 0:
                col = 0

            if cursor == 0 and not line_str:
                if col > 0:
                    line_str += " " * col
                    cursor = col
            else:
                needed = col - cursor
                if needed <= 0:
                    needed = 1
                line_str += " " * needed
                cursor += needed

            line_str += w["text"]
            cursor += len(w["text"])

        out_lines.append(line_str)

    return "\n".join(out_lines)


# ==================== OCR ====================
config = build_config(
    args.oem,
    args.psm,
    config_flags,
    args.dpi,
    args.tessdata_dir,
    args.user_words,
    args.user_patterns,
)

if "DOCS" not in globals():
    DOCS = []


def _basename(val) -> Optional[str]:
    if val is None:
        return None
    try:
        return Path(val).name
    except Exception:
        s = str(val)
        return s.replace("\\", "/").split("/")[-1]


# Si DOCS est une liste de "pages" legacy, on pack en docs
if DOCS and isinstance(DOCS[0], dict) and "pages" not in DOCS[0]:
    groups = {}
    for i, page in enumerate(DOCS, start=1):
        raw = str(page.get("path") or "batch")
        key = f"{raw}::p{page.get('page_index') or i}"
        groups.setdefault(key, []).append(page)

    packed = []
    for key, pages in groups.items():
        pages_sorted = sorted(pages, key=lambda p: int(p.get("page_index") or 0)) if pages else []

        source_files = [_basename(p.get("path")) for p in pages_sorted if _basename(p.get("path"))]
        source_files = list(dict.fromkeys(source_files))

        filename = source_files[0] if len(source_files) == 1 else (_basename(key) or "batch")

        doc = {"doc_id": str(uuid.uuid4()), "filename": filename, "source_files": source_files, "pages": []}
        page_index = 1
        for p in pages_sorted:
            idx = int(p.get("page_index") or page_index)
            src_path = p.get("path")
            doc["pages"].append({
                "page_index": idx,
                "image": p.get("original"),
                "prepped": p.get("prepped"),
                "source_path": src_path,
                "source_file": _basename(src_path)
            })
            page_index += 1
        doc["page_count_total"] = len(doc["pages"])
        packed.append(doc)

    DOCS = packed


# Consistance metadata doc/pages
for doc in DOCS:
    pages = doc.get("pages", []) or []
    for i, page in enumerate(pages, start=1):
        if not page.get("page_index"):
            page["page_index"] = i
        if not page.get("source_file"):
            src_path = page.get("source_path") or page.get("path")
            page["source_file"] = _basename(src_path)

    doc["page_count_total"] = len(pages)

    if not doc.get("source_files"):
        source_files = [p.get("source_file") for p in pages if p.get("source_file")]
        doc["source_files"] = list(dict.fromkeys(source_files))

    if not doc.get("filename"):
        if len(doc.get("source_files", [])) == 1:
            doc["filename"] = doc["source_files"][0]
        elif len(doc.get("source_files", [])) > 1:
            doc["filename"] = "batch"


# OCR chaque page prepped
for doc in DOCS:
    pages_text = []
    for page in doc.get("pages", []):
        prepped = page.get("prepped")
        if prepped is None:
            raise RuntimeError("prepped image missing. Run the input/preprocess step first.")

        data = pytesseract.image_to_data(prepped, lang=args.lang, config=config, output_type=Output.DICT)
        w, h = prepped.size
        ocr_text = _render_layout_from_data(data, w, h)

        page["ocr_text"] = ocr_text
        pages_text.append(ocr_text)

        if PRINT_DURING_EXTRACTION:
            src = page.get("source_file") or _basename(page.get("source_path")) or ""
            total = doc.get("page_count_total", 1)
            print(f"[ocr] {doc.get('filename')} | file={src} | page {page.get('page_index')}/{total}")
            print(ocr_text)
            print("-" * 120)

    doc["pages_text"] = pages_text
    doc["ocr_text"] = "\n\n".join(pages_text)


# ==================== EXTRACTION NATIVE (TEXT_FILES) ====================
def _get_pdf_reader_with_name():
    try:
        from pypdf import PdfReader  # type: ignore
        return PdfReader, "pypdf"
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            return PdfReader, "PyPDF2"
        except ImportError:
            return None, "none"


def _pdf_extract_text_preserve_layout(page) -> str:
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        return page.extract_text() or ""
    except Exception:
        try:
            return page.extract_text() or ""
        except Exception:
            return ""


def _docx_xml_to_text(xml_bytes: bytes) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ET.fromstring(xml_bytes)

    out_lines = []
    for p in root.findall(".//w:p", ns):
        line_parts = []
        for node in p.iter():
            tag = node.tag
            if tag.endswith("}t"):
                line_parts.append(node.text if node.text is not None else "")
            elif tag.endswith("}tab"):
                line_parts.append("\t")
            elif tag.endswith("}br") or tag.endswith("}cr"):
                line_parts.append("\n")
        out_lines.append("".join(line_parts))
    return "\n".join(out_lines)


def _pptx_slide_xml_to_text(xml_bytes: bytes) -> str:
    ns = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    }
    root = ET.fromstring(xml_bytes)

    out_lines = []
    for para in root.findall(".//a:p", ns):
        parts = []
        for node in para.iter():
            tag = node.tag
            if tag.endswith("}t"):
                parts.append(node.text if node.text is not None else "")
            elif tag.endswith("}br"):
                parts.append("\n")
        out_lines.append("".join(parts))
    return "\n".join(out_lines)


def _xlsx_col_to_index(col_letters: str) -> int:
    n = 0
    for ch in col_letters:
        if "A" <= ch <= "Z":
            n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def _xlsx_shared_strings(xml_bytes: bytes) -> list:
    root = ET.fromstring(xml_bytes)
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    out = []
    for si in root.findall(".//s:si", ns):
        parts = []
        for t in si.findall(".//s:t", ns):
            parts.append(t.text if t.text is not None else "")
        out.append("".join(parts))
    return out


def _xlsx_sheet_to_text(sheet_xml: bytes, shared: list) -> str:
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(sheet_xml)

    lines = []
    for row in root.findall(".//s:row", ns):
        cells = row.findall("./s:c", ns)
        row_map = {}
        max_col = 0

        for c in cells:
            r = c.get("r") or ""
            col_letters = "".join([ch for ch in r if ch.isalpha()]).upper()
            col_idx = _xlsx_col_to_index(col_letters) if col_letters else 0
            if col_idx > max_col:
                max_col = col_idx

            cell_type = c.get("t")
            v = c.find("./s:v", ns)
            is_node = c.find("./s:is", ns)

            val = ""
            if cell_type == "s" and v is not None and v.text is not None:
                try:
                    val = shared[int(v.text)]
                except Exception:
                    val = v.text
            elif cell_type == "inlineStr" and is_node is not None:
                parts = []
                for t in is_node.findall(".//s:t", ns):
                    parts.append(t.text if t.text is not None else "")
                val = "".join(parts)
            else:
                if v is not None and v.text is not None:
                    val = v.text

            row_map[col_idx] = val

        if max_col <= 0:
            lines.append("")
        else:
            parts = []
            for i in range(1, max_col + 1):
                parts.append(row_map.get(i, ""))
            # IMPORTANT: séparateur \t pour préserver colonnes
            lines.append("\t".join(parts))

    return "\n".join(lines)


def _odf_content_to_text(xml_bytes: bytes) -> str:
    ns_text = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    root = ET.fromstring(xml_bytes)

    def walk(node):
        pieces = []
        if node.text is not None:
            pieces.append(node.text)

        for child in list(node):
            tag = child.tag
            if tag == f"{{{ns_text}}}s":
                c = child.get(f"{{{ns_text}}}c") or child.get("c") or "1"
                try:
                    pieces.append(" " * int(c))
                except Exception:
                    pieces.append(" ")
            else:
                pieces.append(walk(child))

            if child.tail is not None:
                pieces.append(child.tail)
        return "".join(pieces)

    out_lines = []
    for p in root.iter():
        if p.tag == f"{{{ns_text}}}p":
            out_lines.append(walk(p))
    return "\n".join(out_lines)


def _html_bytes_to_text_preserve(b: bytes) -> str:
    # Remplacer <br> et </p> par des \n avant de supprimer les tags
    b = re.sub(rb"(?i)<br\s*/?>", b"\n", b)
    b = re.sub(rb"(?i)</p\s*>", b"\n", b)
    b = re.sub(rb"<[^>]+>", b" ", b)
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return str(b)


def extract_text_native(path: str) -> dict:
    ft = detect_path_type(path)
    ext = ft.ext.lower()
    filename = Path(path).name

    # PDF
    if ext == ".pdf":
        PdfReader, backend = _get_pdf_reader_with_name()
        if PdfReader is not None:
            reader = PdfReader(path)
            pages = reader.pages
            pages_text = []
            total = len(pages)

            for i, page in enumerate(pages, start=1):
                txt = _pdf_extract_text_preserve_layout(page)
                pages_text.append(txt)
                if PRINT_DURING_EXTRACTION:
                    print(f"[native:{backend}] {filename} page {i}/{total}")
                    print(txt)
                    print("-" * 120)

            full = "\n\n".join(pages_text)
            return {
                "doc_id": str(uuid.uuid4()),
                "filename": filename,
                "source_path": path,
                "content": "text",
                "extraction": f"native:pdf:{backend}",
                "text": full,
                "pages_text": pages_text,
                "page_count_total": total,
            }

        # Fallback pdfminer
        try:
            from pdfminer.high_level import extract_text  # type: ignore
            full = extract_text(path) or ""
            pages = full.split("\f")
            pages_text = [p for p in pages]  # garder brut, sans strip
            total = len(pages_text)

            if PRINT_DURING_EXTRACTION:
                for i, txt in enumerate(pages_text, start=1):
                    print(f"[native:pdfminer] {filename} page {i}/{total}")
                    print(txt)
                    print("-" * 120)

            full2 = "\n\n".join(pages_text)
            return {
                "doc_id": str(uuid.uuid4()),
                "filename": filename,
                "source_path": path,
                "content": "text",
                "extraction": "native:pdf:pdfminer",
                "text": full2,
                "pages_text": pages_text,
                "page_count_total": total,
            }
        except Exception:
            return {
                "doc_id": str(uuid.uuid4()),
                "filename": filename,
                "source_path": path,
                "content": "text",
                "extraction": "native:pdf:none",
                "text": "",
                "pages_text": [""],
                "page_count_total": 1,
            }

    # Office/OpenDocument/EPUB
    if ext in {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub"}:
        try:
            with zipfile.ZipFile(path, "r") as z:
                names = z.namelist()

                if ext == ".docx":
                    parts = []
                    if "word/document.xml" in names:
                        parts.append(_docx_xml_to_text(z.read("word/document.xml")))
                    for nm in names:
                        if nm.startswith("word/header") and nm.endswith(".xml"):
                            parts.append(_docx_xml_to_text(z.read(nm)))
                        if nm.startswith("word/footer") and nm.endswith(".xml"):
                            parts.append(_docx_xml_to_text(z.read(nm)))
                    text = "\n\n".join(parts)
                    return {
                        "doc_id": str(uuid.uuid4()),
                        "filename": filename,
                        "source_path": path,
                        "content": "text",
                        "extraction": "native:docx:xml",
                        "text": text,
                        "pages_text": [text],
                        "page_count_total": 1,
                    }

                if ext == ".xlsx":
                    shared = []
                    if "xl/sharedStrings.xml" in names:
                        try:
                            shared = _xlsx_shared_strings(z.read("xl/sharedStrings.xml"))
                        except Exception:
                            shared = []

                    sheet_files = [nm for nm in names if nm.startswith("xl/worksheets/") and nm.endswith(".xml")]
                    sheet_files_sorted = sorted(sheet_files)

                    pages_text = []
                    total = len(sheet_files_sorted) or 1
                    for nm in sheet_files_sorted:
                        sheet_text = _xlsx_sheet_to_text(z.read(nm), shared)
                        pages_text.append(sheet_text)

                    text = "\n\n".join(pages_text)
                    return {
                        "doc_id": str(uuid.uuid4()),
                        "filename": filename,
                        "source_path": path,
                        "content": "text",
                        "extraction": "native:xlsx:xml",
                        "text": text,
                        "pages_text": pages_text,
                        "page_count_total": total,
                    }

                if ext == ".pptx":
                    slides = [nm for nm in names if nm.startswith("ppt/slides/") and nm.endswith(".xml")]
                    slides_sorted = sorted(slides)
                    pages_text = []
                    total = len(slides_sorted) or 1
                    for nm in slides_sorted:
                        pages_text.append(_pptx_slide_xml_to_text(z.read(nm)))
                    text = "\n\n".join(pages_text)
                    return {
                        "doc_id": str(uuid.uuid4()),
                        "filename": filename,
                        "source_path": path,
                        "content": "text",
                        "extraction": "native:pptx:xml",
                        "text": text,
                        "pages_text": pages_text,
                        "page_count_total": total,
                    }

                if ext in {".odt", ".ods", ".odp"}:
                    text = ""
                    if "content.xml" in names:
                        text = _odf_content_to_text(z.read("content.xml"))
                    return {
                        "doc_id": str(uuid.uuid4()),
                        "filename": filename,
                        "source_path": path,
                        "content": "text",
                        "extraction": f"native:{ext[1:]}:xml",
                        "text": text,
                        "pages_text": [text],
                        "page_count_total": 1,
                    }

                if ext == ".epub":
                    htmls = [nm for nm in names if nm.lower().endswith((".xhtml", ".html", ".htm"))]
                    htmls_sorted = sorted(htmls)
                    pages_text = []
                    total = len(htmls_sorted) or 1
                    for nm in htmls_sorted:
                        try:
                            b = z.read(nm)
                        except Exception:
                            b = b""
                        pages_text.append(_html_bytes_to_text_preserve(b))
                    text = "\n\n".join(pages_text)
                    return {
                        "doc_id": str(uuid.uuid4()),
                        "filename": filename,
                        "source_path": path,
                        "content": "text",
                        "extraction": "native:epub:html",
                        "text": text,
                        "pages_text": pages_text,
                        "page_count_total": total,
                    }

        except Exception as e:
            return {
                "doc_id": str(uuid.uuid4()),
                "filename": filename,
                "source_path": path,
                "content": "text",
                "extraction": "native:zip:error",
                "text": "",
                "pages_text": [""],
                "page_count_total": 1,
                "error": str(e),
            }

    return {
        "doc_id": str(uuid.uuid4()),
        "filename": filename,
        "source_path": path,
        "content": "text",
        "extraction": "native:unsupported",
        "text": "",
        "pages_text": [""],
        "page_count_total": 1,
    }


# ==================== Collecte TEXT_FILES ====================
TEXT_DOCS: List[dict] = []
if "TEXT_FILES" not in globals():
    TEXT_FILES = []

for p in TEXT_FILES:
    try:
        TEXT_DOCS.append(extract_text_native(p))
    except Exception as e:
        TEXT_DOCS.append({
            "doc_id": str(uuid.uuid4()),
            "filename": Path(p).name,
            "source_path": p,
            "content": "text",
            "extraction": "native:error",
            "text": "",
            "pages_text": [""],
            "page_count_total": 1,
            "error": str(e),
        })


# ==================== Sortie finale (OCR + Native) ====================
FINAL_DOCS: List[dict] = []

# OCR docs (images)
for d in DOCS:
    pages_text = d.get("pages_text") or []
    page_count_total = d.get("page_count_total") or len(pages_text) or 1
    FINAL_DOCS.append({
        "doc_id": d.get("doc_id"),
        "filename": d.get("filename"),
        "content": "image_only",
        "extraction": "ocr:tesseract",
        "text": d.get("ocr_text", ""),
        "pages_text": pages_text,
        "page_count_total": page_count_total,
    })

# Native docs (text)
for d in TEXT_DOCS:
    pages_text = d.get("pages_text") or [d.get("text") or ""]
    page_count_total = d.get("page_count_total") or len(pages_text) or 1
    FINAL_DOCS.append({
        "doc_id": d.get("doc_id"),
        "filename": d.get("filename"),
        "content": "text",
        "extraction": d.get("extraction"),
        "text": d.get("text", ""),
        "pages_text": pages_text,
        "page_count_total": page_count_total,
    })


# ==================== Print audit (préserver espaces) ====================
for d in FINAL_DOCS:
    filename = d.get("filename")
    content = d.get("content")
    extraction = d.get("extraction")
    pages_text = d.get("pages_text") or []
    total = int(d.get("page_count_total") or len(pages_text) or 1)

    print(f"[doc] {filename} | content={content} | extraction={extraction} | pages={total}")

    if not pages_text:
        print()
        print("-" * 120)
        print()
        continue

    for i, txt in enumerate(pages_text, start=1):
        print(f"[page {i}/{total}]")

        # IMPORTANT: on imprime sans transformer la chaîne
        # end="" pour ne pas ajouter de newline "surprise" (on la gère nous-même)
        if txt is None:
            txt = ""
        print(txt, end="")

        # Assurer séparation propre: si txt ne finit pas par \n, on ajoute une newline
        if not txt.endswith("\n"):
            print()

        print("-" * 120)

    print()
