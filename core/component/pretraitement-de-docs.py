from __future__ import annotations

import csv
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Sequence, Union, List, Dict, Any


# Saisie possible:
# INPUT_FILE = "a.pdf, b.docx, c.png"
# INPUT_FILE = ["a.pdf", "b.docx", "c.png"]
INPUT_FILE: Optional[Union[str, Sequence[str]]] = (
    # "epsteanpdf.pdf, epsteain22.pdf, testexcel.xlsx, testword.docx, image2tab.webp, contras-14page.pdf, signettab.png"
    # "contras-14page.pdf, testword.docx, testexcel.xlsx, signettab.png, image2tab.webp"
    "documents/testword.docx"
)

# Heuristiques
MIN_CHARS_OFFICE = 1     # 1 caractère => "text"
MIN_CHARS_PDF = 30       # seuil de texte extrait
PDF_MAX_PAGES = 3        # on teste les N premières pages

# Dossiers de recherche si un nom est donné sans chemin (utile en notebook)
SEARCH_DIRS = [
    os.getcwd(),
    "/mnt/data",  # utile dans l'environnement ChatGPT
]


@dataclass(frozen=True)
class FileType:
    ext: str
    mime: str
    label: str


# ----------------- input parsing -----------------

def normalize_input_files(x: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Retourne toujours une liste. Supporte une string avec virgules (CSV)."""
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if "," not in s:
            return [s]
        parts = next(csv.reader([s], skipinitialspace=True))
        return [p.strip() for p in parts if p.strip()]
    return [str(p).strip() for p in x if str(p).strip()]


def resolve_path(p: str) -> Optional[str]:
    """
    Résout un chemin:
    - si p existe tel quel -> retourne p
    - sinon essaie SEARCH_DIRS + basename(p)
    - sinon retourne None (introuvable)
    """
    p = os.path.expandvars(os.path.expanduser(p.strip()))
    if os.path.exists(p):
        return p

    base = os.path.basename(p)
    for d in SEARCH_DIRS:
        alt = os.path.join(d, base)
        if os.path.exists(alt):
            return alt

    return None


# ----------------- format detection -----------------

def _read_head(path: str, n: int = 16384) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def detect_path_type(path: str) -> FileType:
    head = _read_head(path)

    if head.startswith(b"%PDF-"):
        return FileType(".pdf", "application/pdf", "PDF document")

    if head.startswith(b"II*\x00") or head.startswith(b"MM\x00*"):
        return FileType(".tif", "image/tiff", "TIFF image")

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return FileType(".png", "image/png", "PNG image")

    if head.startswith(b"\xff\xd8\xff"):
        return FileType(".jpg", "image/jpeg", "JPEG image")

    if len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return FileType(".webp", "image/webp", "WEBP image")

    # ZIP containers (DOCX/XLSX/PPTX/ODT/ODS/ODP/EPUB/ZIP)
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        try:
            with zipfile.ZipFile(path, "r") as z:
                names = set(z.namelist())

                # EPUB
                if "mimetype" in names and "META-INF/container.xml" in names:
                    try:
                        mt = z.read("mimetype")[:64].decode("ascii", errors="ignore").strip()
                    except Exception:
                        mt = ""
                    if mt == "application/epub+zip":
                        return FileType(".epub", "application/epub+zip", "EPUB eBook")

                # Office OpenXML
                if "word/document.xml" in names:
                    return FileType(".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Word document (DOCX)")
                if "xl/workbook.xml" in names:
                    return FileType(".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "Excel workbook (XLSX)")
                if "ppt/presentation.xml" in names:
                    return FileType(".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", "PowerPoint presentation (PPTX)")

                # OpenDocument
                if "content.xml" in names and "META-INF/manifest.xml" in names:
                    mt = ""
                    try:
                        if "mimetype" in names:
                            mt = z.read("mimetype")[:128].decode("ascii", errors="ignore").strip()
                    except Exception:
                        mt = ""
                    if mt == "application/vnd.oasis.opendocument.text":
                        return FileType(".odt", mt, "OpenDocument Text (ODT)")
                    if mt == "application/vnd.oasis.opendocument.spreadsheet":
                        return FileType(".ods", mt, "OpenDocument Spreadsheet (ODS)")
                    if mt == "application/vnd.oasis.opendocument.presentation":
                        return FileType(".odp", mt, "OpenDocument Presentation (ODP)")
                    return FileType(".odf", "application/zip", "OpenDocument container")

        except Exception:
            pass

        return FileType(".zip", "application/zip", "ZIP archive/container")

    # Ancien Office (OLE2)
    if head.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return FileType(".ole", "application/x-ole-storage", "OLE2 container (old Office)")

    return FileType("", "application/octet-stream", "Unknown / binary")


# ----------------- text vs image_only -----------------

def _xml_text_len(xml_bytes: bytes) -> int:
    """Compte du texte dans du XML (éléments + fallback simple)."""
    try:
        root = ET.fromstring(xml_bytes)
        total = 0
        for elem in root.iter():
            if elem.text and elem.text.strip():
                total += len(elem.text.strip())
        return total
    except Exception:
        s = re.sub(rb"<[^>]+>", b" ", xml_bytes)
        return len(re.sub(rb"\s+", b" ", s).strip())


def _zip_has_text(path: str, ext: str) -> bool:
    """
    DOCX/XLSX/PPTX/ODT/ODS/ODP/EPUB
    True si on trouve au moins MIN_CHARS_OFFICE caractères.
    """
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()

            if ext == ".docx":
                total = 0
                # corps
                if "word/document.xml" in names:
                    total += _xml_text_len(z.read("word/document.xml"))
                # headers/footers (souvent du texte “isolé”)
                for nm in names:
                    if nm.startswith("word/header") and nm.endswith(".xml"):
                        total += _xml_text_len(z.read(nm))
                    if nm.startswith("word/footer") and nm.endswith(".xml"):
                        total += _xml_text_len(z.read(nm))
                    if total >= MIN_CHARS_OFFICE:
                        break
                return total >= MIN_CHARS_OFFICE

            if ext == ".xlsx":
                total = 0
                if "xl/sharedStrings.xml" in names:
                    total += _xml_text_len(z.read("xl/sharedStrings.xml"))
                if total < MIN_CHARS_OFFICE:
                    for nm in names:
                        if nm.startswith("xl/worksheets/") and nm.endswith(".xml"):
                            total += _xml_text_len(z.read(nm))
                            if total >= MIN_CHARS_OFFICE:
                                break
                return total >= MIN_CHARS_OFFICE

            if ext == ".pptx":
                total = 0
                for nm in names:
                    if nm.startswith("ppt/slides/") and nm.endswith(".xml"):
                        total += _xml_text_len(z.read(nm))
                        if total >= MIN_CHARS_OFFICE:
                            break
                return total >= MIN_CHARS_OFFICE

            if ext in {".odt", ".ods", ".odp"}:
                if "content.xml" in names:
                    return _xml_text_len(z.read("content.xml")) >= MIN_CHARS_OFFICE
                return False

            if ext == ".epub":
                total = 0
                for nm in names:
                    low = nm.lower()
                    if low.endswith((".xhtml", ".html", ".htm")):
                        try:
                            b = z.read(nm)
                        except Exception:
                            continue
                        s = re.sub(rb"<[^>]+>", b" ", b)
                        total += len(re.sub(rb"\s+", b" ", s).strip())
                        if total >= MIN_CHARS_OFFICE:
                            break
                return total >= MIN_CHARS_OFFICE

    except Exception:
        return False

    return False


def _get_pdf_reader():
    """Retourne PdfReader depuis pypdf ou PyPDF2, ou None si indisponible."""
    try:
        from pypdf import PdfReader  # type: ignore
        return PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            return PdfReader
        except ImportError:
            return None


def _pdf_has_text(path: str) -> bool:
    """
    PDF:
    - True si extract_text() produit assez de caractères, OU si fonts / opérateurs texte présents.
    - Si aucune lib PDF n'est dispo: fallback binaire (cherche /Font ou opérateurs BT/Tj).
    """
    PdfReader = _get_pdf_reader()
    if PdfReader is None:
        # fallback binaire: moins fiable, mais évite de renvoyer faux systématique
        try:
            with open(path, "rb") as f:
                data = f.read(2_000_000)  # 2MB max
            if b"/Font" in data:
                return True
            if b"BT" in data and (b"Tj" in data or b"TJ" in data):
                return True
        except Exception:
            pass
        return False

    try:
        reader = PdfReader(path)
        pages = reader.pages[: max(1, PDF_MAX_PAGES)]

        extracted_score = 0
        saw_font = False
        saw_text_ops = False

        for page in pages:
            # 1) extraction texte
            txt = page.extract_text() or ""
            extracted_score += len("".join(txt.split()))
            if extracted_score >= MIN_CHARS_PDF:
                return True

            # 2) fonts dans resources
            try:
                res = page.get("/Resources") or {}
                font = res.get("/Font")
                if font:
                    saw_font = True
            except Exception:
                pass

            # 3) opérateurs texte dans stream
            try:
                contents = page.get_contents()
                if contents is None:
                    continue
                if hasattr(contents, "get_data"):
                    data = contents.get_data()
                else:
                    data = b"".join(c.get_data() for c in contents)  # type: ignore
                if b"BT" in data and (b"Tj" in data or b"TJ" in data):
                    saw_text_ops = True
            except Exception:
                pass

        return saw_font or saw_text_ops

    except Exception:
        return False


def content_kind_two_states(path: str, ftype: FileType) -> str:
    """Retourne seulement: 'text' ou 'image_only'."""
    ext = ftype.ext.lower()

    # Images => image_only
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".ico"}:
        return "image_only"

    # PDF
    if ext == ".pdf":
        return "text" if _pdf_has_text(path) else "image_only"

    # Formats texte compressés (Office/ODF/EPUB)
    if ext in {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub"}:
        return "text" if _zip_has_text(path, ext) else "image_only"

    # Tout le reste => image_only (car tu veux 2 états)
    return "image_only"


def analyze_many_two_states(input_file: Optional[Union[str, Sequence[str]]]) -> List[Dict[str, Any]]:
    """
    Sortie:
      [{"path": ..., "ext": ..., "mime": ..., "label": ..., "content": "text|image_only"}, ...]
    Ignore les fichiers introuvables.
    """
    raw_paths = normalize_input_files(input_file)
    out: List[Dict[str, Any]] = []

    for raw in raw_paths:
        p = resolve_path(raw)
        if p is None:
            continue

        ft = detect_path_type(p)
        out.append({
            "path": p,
            "ext": ft.ext,
            "mime": ft.mime,
            "label": ft.label,
            "content": content_kind_two_states(p, ft),
        })

    return out


# Test
analyze_many_two_states(INPUT_FILE)
