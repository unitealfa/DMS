import re
import pickle
import math
from pathlib import Path
import nltk

# ==================== Fallbacks (prod-safe, sans changer la logique du test) ====================
# IMPORTANT:
# - On ne "fabrique" PAS FINAL_DOCS/DOCS/TEXT_DOCS si absents, sinon a masque l'erreur et la logique diverge.
# - On garde juste un fallback pour _get_pdf_reader (optionnel) pour excution standalone.
if "_get_pdf_reader" not in globals():
    def _get_pdf_reader():
        return None
if "FINAL_DOCS" not in globals():
    FINAL_DOCS = None  # type: ignore
if "DOCS" not in globals():
    DOCS = None  # type: ignore
if "TEXT_DOCS" not in globals():
    TEXT_DOCS = None  # type: ignore

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ==================== Reglages ====================
TARGET = None

PRINT_SENTENCES = True
MAX_SENTENCES_PREVIEW = 80   # None => imprime tout
PRINT_REPR = False           # True => debug espaces invisibles via repr(chunk)

MIN_SENTENCE_NONSPACE = 12
PRINT_ONLY_SENTENCES = True
PRINT_PAGE_TEXT = False

# ==================== NLTK data ====================
def _ensure_nltk():
    for pkg, probe in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(probe)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                print(f"[warn] NLTK download failed for {pkg}: {e}")

_ensure_nltk()

# ==================== Dtection langue (simple) ====================
_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]+", flags=re.UNICODE)

_FR_HINT = {"le","la","les","des","une","un","est","avec","pour","dans","sur","facture","date","total","tva","montant"}
_EN_HINT = {"the","and","to","of","in","is","for","with","invoice","date","total","vat","amount"}

def detect_lang(text: str) -> str:
    t = text or ""
    if _AR_RE.search(t):
        return "ar"
    words = [w.lower() for w in _WORD_RE.findall(t[:8000])]
    if not words:
        return "en"
    fr_score = sum(1 for w in words if w in _FR_HINT)
    en_score = sum(1 for w in words if w in _EN_HINT)
    # Accent hint for French
    if re.search(r"[\u00E8\u00E9\u00EA\u00EB\u00E0\u00F9\u00E7\u00F4\u00EE\u00EF]", t.lower()):
        fr_score += 1
    return "fr" if fr_score >= en_score else "en"

# ==================== Sentence split "layout" (fallback) ====================
_AR_END_RE = re.compile(r"([.!?]+)(\s+|$)", flags=re.UNICODE)

def split_ar_layout(text: str):
    if not text:
        return []
    chunks = []
    last = 0
    for m in _AR_END_RE.finditer(text):
        end = m.end()
        chunks.append(text[last:end])
        last = end
    if last < len(text):
        chunks.append(text[last:])
    return chunks

def _load_punkt_pickle(lang_pickle_name: str):
    p = nltk.data.find(f"tokenizers/punkt/{lang_pickle_name}.pickle")
    with open(p, "rb") as f:
        return pickle.load(f)

def split_punkt_layout(text: str, lang_pickle_name: str):
    if not text:
        return []
    tok = _load_punkt_pickle(lang_pickle_name)
    spans = list(tok.span_tokenize(text))
    if not spans:
        return [text]
    starts = [0] + [spans[i][0] for i in range(1, len(spans))]
    ends = [spans[i+1][0] for i in range(len(spans)-1)] + [len(text)]
    return [text[starts[i]:ends[i]] for i in range(len(ends))]

def sentence_chunks_layout(text: str, lang: str):
    lang = (lang or "").lower()
    if lang.startswith("ar"):
        return split_ar_layout(text)
    if lang.startswith("fr"):
        return split_punkt_layout(text, "french")
    if lang.startswith("en"):
        return split_punkt_layout(text, "english")
    return split_punkt_layout(text, "english")

# ==================== Split sections/alinas (layout-preserving) ====================
def _iter_line_spans(text: str):
    if not text:
        return
    start = 0
    for m in re.finditer(r"\n", text):
        end = m.end()
        yield start, end
        start = end
    if start < len(text):
        yield start, len(text)

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _mask_digits(s: str) -> str:
    return re.sub(r"\d", "#", s)

_NUM_SIMPLE_RE = re.compile(r"(?i)^[ \t]*\(?\d{1,3}\)?[ \t]*[.)][ \t]*(?:\S|$)")
_ALPHA_RE      = re.compile(r"(?i)^[ \t]*\(?[a-z]\)?[ \t]*[.)][ \t]*(?:\S|$)")
_ROMAN_RE      = re.compile(r"(?i)^[ \t]*\(?[ivxlcdm]{1,8}\)?[ \t]*[.)][ \t]*(?:\S|$)")
_NUM_MULTI_RE  = re.compile(r"^[ \t]*\d{1,3}(?:\.\d{1,3})+[ \t]*(?:[.)])?[ \t]+(?=\S)")

_KEYWORD_STRONG_RE = re.compile(r"(?i)^[ \t]*(article|section|chapitre|chapter|part)\b")
_KEYWORD_WEAK_HEADING_RE = re.compile(
    r"""(?ix)^[ \t]*
    (schedule|exhibit|appendix|annexe|annex)
    [ \t]+
    ([A-Z0-9]{1,8}|[ivxlcdm]{1,8}|\d{1,3})
    [ \t]*
    (?:[:\-][ \t]*\S.*)?
    [ \t]*$
    """
)
_SEP_RE = re.compile(r"^[ \t]*[-_]{4,}[ \t]*$")

_LABEL_ONLY_RE = re.compile(
    r"(?is)^[ \t]*"
    r"(?:\(?\d{1,3}\)?|\(?[a-z]\)?|\(?[ivxlcdm]{1,8}\)?)"
    r"[ \t]*[.)][ \t]*$"
)

def _is_section_start_line(line: str) -> bool:
    s = (line or "").rstrip("\n")
    st = s.strip()
    if not st:
        return False
    if _SEP_RE.match(st):
        return False
    if _KEYWORD_STRONG_RE.match(s):
        return True
    if _KEYWORD_WEAK_HEADING_RE.match(s):
        return True
    if _NUM_SIMPLE_RE.match(s):
        return True
    if _NUM_MULTI_RE.match(s):
        label = _collapse_ws(s).split(" ", 1)[0]
        parts = label.split(".")
        if len(parts) >= 2 and parts[-1] in ("00", "000"):
            return False
        return True
    if _ALPHA_RE.match(s) or _ROMAN_RE.match(s):
        return True
    return False

def _merge_label_only(chunks):
    out = []
    i = 0
    while i < len(chunks):
        if i + 1 < len(chunks) and _LABEL_ONLY_RE.match(chunks[i]):
            out.append(chunks[i] + chunks[i+1])
            i += 2
        else:
            out.append(chunks[i])
            i += 1
    return out

def split_sections_layout(text: str, allow_alpha_roman: bool = True):
    if not text:
        return []
    starts = {0}
    for ls, le in _iter_line_spans(text):
        line = text[ls:le]
        if _is_section_start_line(line):
            if not allow_alpha_roman:
                s = line.rstrip("\n")
                if (
                    _KEYWORD_STRONG_RE.match(s)
                    or _KEYWORD_WEAK_HEADING_RE.match(s)
                    or _NUM_SIMPLE_RE.match(s)
                    or _NUM_MULTI_RE.match(s)
                ):
                    starts.add(ls)
            else:
                starts.add(ls)

    starts = sorted(starts)
    if len(starts) == 1:
        return [text]

    chunks = []
    for i in range(len(starts) - 1):
        a, b = starts[i], starts[i+1]
        if a != b:
            chunks.append(text[a:b])
    chunks.append(text[starts[-1]:])

    return _merge_label_only(chunks)

_PARA_BREAK_RE = re.compile(r"(?:\n[ \t]*){2,}")

def split_paragraphs_layout(text: str):
    if not text:
        return []
    starts = [0]
    for m in _PARA_BREAK_RE.finditer(text):
        starts.append(m.end())
    starts = sorted(set(starts))
    if len(starts) == 1:
        return [text]
    out = []
    for i in range(len(starts) - 1):
        out.append(text[starts[i]:starts[i+1]])
    out.append(text[starts[-1]:])
    return out

def chunk_layout_universal(text: str, lang: str):
    if not text:
        return []

    lines = [text[ls:le].rstrip("\n") for ls, le in _iter_line_spans(text)]
    num_kw_hits = 0
    alpha_roman_hits = 0

    for ln in lines:
        if not ln.strip():
            continue
        if (
            _KEYWORD_STRONG_RE.match(ln)
            or _KEYWORD_WEAK_HEADING_RE.match(ln)
            or _NUM_SIMPLE_RE.match(ln)
            or _NUM_MULTI_RE.match(ln)
        ):
            num_kw_hits += 1
        elif _ALPHA_RE.match(ln) or _ROMAN_RE.match(ln):
            alpha_roman_hits += 1

    is_structured = (num_kw_hits >= 2) or (alpha_roman_hits >= 3)

    if is_structured:
        chunks = split_sections_layout(text, allow_alpha_roman=True)
        if len(chunks) > 1:
            return chunks

    paras = split_paragraphs_layout(text)
    if len(paras) > 1:
        return paras

    return sentence_chunks_layout(text, lang)

# ======================================================================
#  MULTI-COLONNES (gnral, robuste) + TABLE (inchang)
#  + micro-table: interprter les headers multi-col comme un "table chunk"
# ======================================================================

GAP_MIN_OCR = 10
GAP_MIN_NATIVE = 6

MERGE_COL_DIST_OCR = 22
MERGE_COL_DIST_NATIVE = 16

MICROTABLE_MAX_ROWS = 30
MICROTABLE_MIN_DENS = 0.25
MICROTABLE_MIN_MULTIROW = 2

TABLE_HINT_RE = re.compile(
    r"""(?ix)
    \b(
        qt[e]|
        dsignation|designation|
        prix|
        montan?t|
        r[e]f[e]rence|reference|
        description|
        quantit[e]|
        p\.?\s*unitaire|
        valeur|
        total\s*ht|total|
        tva|vat
    )\b
    """
)

NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
DEC_RE = re.compile(r"\d+[.,]\d+")

def _space_runs_ge(s: str, n: int):
    return [(m.start(), m.end()) for m in re.finditer(r"[ ]{%d,}" % n, s or "")]

def _has_big_gap(s: str, gap_min: int, min_count: int = 1) -> bool:
    return len(_space_runs_ge(s, gap_min)) >= min_count

def _num_tokens(s: str) -> int:
    return len(NUM_RE.findall(s or ""))

def _dec_tokens(s: str) -> int:
    return len(DEC_RE.findall(s or ""))

def _is_table_line(line: str, gap_min: int) -> bool:
    s = (line or "").rstrip("\n")
    if not s.strip():
        return False
    if s.count("\t") >= 2:
        return True
    if TABLE_HINT_RE.search(s):
        return True
    if _has_big_gap(s, gap_min, min_count=2):
        if _num_tokens(s) >= 3:
            return True
        if _dec_tokens(s) >= 1:
            return True
    return False

def _cluster_centers(values, tol=2, min_hits=1):
    if not values:
        return []
    xs = sorted(values)
    clusters = []
    cur = [xs[0]]
    for v in xs[1:]:
        if abs(v - cur[-1]) <= tol:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)

    centers = []
    for c in clusters:
        if len(c) >= min_hits:
            c2 = sorted(c)
            centers.append(c2[len(c2)//2])
    return sorted(set(centers))

def _upper_ratio(s: str) -> float:
    letters = re.findall(r"[A-Za-z---]", s or "")
    if not letters:
        return 0.0
    upp = sum(1 for ch in letters if ch.isupper())
    return upp / max(1, len(letters))

def _sep_spans(line: str, gap_min: int):
    s = line or ""
    spans = []
    for m in re.finditer(r"\t+", s):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r"[ ]{%d,}" % gap_min, s):
        spans.append((m.start(), m.end()))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged

def _line_segments_by_gaps(line: str, gap_min: int):
    s = (line or "").rstrip("\n")
    if not s.strip():
        return []
    seps = _sep_spans(s, gap_min)
    segs = []
    prev = 0
    cuts = seps + [(len(s), len(s))]
    for a, b in cuts:
        if a < prev:
            continue
        chunk = s[prev:a]
        m1 = re.search(r"\S", chunk)
        if m1:
            l = m1.start()
            r = len(chunk.rstrip(" \t"))
            text = chunk[l:r]
            segs.append({"x": prev + l, "a": prev + l, "b": prev + r, "text": text})
        prev = b
    return segs

def _looks_like_title_line(line: str) -> bool:
    s = (line or "").rstrip("\n").strip()
    if not s or len(s) > 50:
        return False
    if _SEP_RE.match(s):
        return False
    if _is_section_start_line(line):
        return True
    if _upper_ratio(s) >= 0.85 and re.search(r"[A-Za-z---]", s) and not re.search(r"\d", s):
        return True
    return False

def _is_multicol_candidate_line(line: str, gap_min: int, is_ocr: bool) -> bool:
    s = (line or "").rstrip("\n")
    st = s.strip()
    if not st:
        return False
    if _SEP_RE.match(st):
        return False
    if _is_table_line(line, gap_min):
        return False

    segs = _line_segments_by_gaps(s, gap_min)
    if len(segs) >= 3:
        return True
    if len(segs) == 2:
        if is_ocr:
            return True
        if _has_big_gap(s, gap_min, 1):
            return True
        if re.search(r"[:#/\\\-]", s) or re.search(r"\d", s):
            return True
        if _upper_ratio(s) >= 0.70:
            return True
    return False

_KV_GENERIC_RE = re.compile(r"^\s*(?P<k>[^:]{1,80}?)\s{2,}(?P<v>\S.+?)\s*$")

def _looks_like_header_pair(k: str, v: str) -> bool:
    k2 = (k or "").strip()
    v2 = (v or "").strip()
    if not k2 or not v2:
        return False
    if len(k2) <= 25 and len(v2) <= 25 and _upper_ratio(k2) >= 0.85 and _upper_ratio(v2) >= 0.85:
        if not re.search(r"\d", k2 + v2):
            return True
    return False

def _looks_like_addressish(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if re.search(r"(rue|route|avenue|bd|boulevard|street|st\.|road|zip|code\s*postal|bp)", s, flags=re.I):
        return True
    if len(s) >= 10 and not s.endswith(":") and re.search(r"[A-Za-z---]", s):
        return True
    return False

def _normalize_kv_generic(text: str) -> str:
    out = []
    for raw in (text or "").splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            out.append("")
            continue
        if ":" in line:
            out.append(line.strip())
            continue
        if _looks_like_addressish(line):
            out.append(line.strip())
            continue
        m = _KV_GENERIC_RE.match(line)
        if not m:
            out.append(line.strip())
            continue
        k = _collapse_ws(m.group("k"))
        v = _collapse_ws(m.group("v"))
        if _looks_like_header_pair(k, v):
            out.append(line.strip())
            continue
        if not re.search(r"[A-Za-z---]", k):
            out.append(line.strip())
            continue
        out.append(f"{k}: {v}" if v else k)
    return "\n".join(out) + ("\n" if (text or "").endswith("\n") else "")

def _strip_sep_lines(block_text: str) -> str:
    if not block_text:
        return ""
    out = []
    for ln in (block_text or "").splitlines():
        if _SEP_RE.match(ln.strip()):
            continue
        out.append(ln.rstrip())
    txt = "\n".join(out).rstrip()
    return txt + ("\n" if (block_text or "").endswith("\n") else "")

def _assign_to_centers(x: int, centers, tol: int):
    if not centers:
        return 0
    best_i = 0
    best_d = abs(x - centers[0])
    for i in range(1, len(centers)):
        d = abs(x - centers[i])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

def _merge_close_columns(centers, row_cells, merge_dist: int):
    i = 0
    while i < len(centers) - 1:
        if (centers[i+1] - centers[i]) <= merge_dist:
            both = 0
            alone_next = 0
            for r in row_cells:
                hi = i in r
                hj = (i+1) in r
                if hj and hi:
                    both += 1
                elif hj and not hi:
                    alone_next += 1
            if both >= 1 and alone_next <= max(1, int(0.2 * (both + alone_next))):
                for r in row_cells:
                    if (i+1) in r:
                        t2, sp2 = r.pop(i+1)
                        if i in r:
                            t1, sp1 = r[i]
                            r[i] = ((t1 + "  " + t2).strip(), sp1 + sp2)
                        else:
                            r[i] = (t2, sp2)

                centers.pop(i+1)

                for r in row_cells:
                    ks = sorted([k for k in r.keys() if k > i+1])
                    for k in ks:
                        r[k-1] = r.pop(k)
                continue
        i += 1
    return centers, row_cells

def _is_grid_like(row_cells, col_count: int):
    if col_count < 2:
        return False
    rows = [r for r in row_cells if any((t.strip() for t, _ in r.values()))]
    if not rows:
        return False
    n_rows = len(rows)
    if n_rows > 5:
        return False
    dens = sum((len(r) / max(1, col_count)) for r in rows) / n_rows
    return dens >= 0.70

def _is_micro_table_like(row_cells, col_count: int) -> bool:
    if col_count < 2:
        return False
    rows = [r for r in row_cells if any((t.strip() for t, _ in r.values()))]
    if len(rows) < 2:
        return False
    if len(rows) > MICROTABLE_MAX_ROWS:
        return False
    multi = sum(1 for r in rows if len(r) >= 2)
    if multi < MICROTABLE_MIN_MULTIROW:
        return False
    dens = sum((len(r) / max(1, col_count)) for r in rows) / max(1, len(rows))
    return dens >= MICROTABLE_MIN_DENS

def _transpose_or_group_multicol(block_text: str, abs_start: int, gap_min: int, is_ocr: bool):
    lines = []
    segs_by_line = []

    for ls, le in _iter_line_spans(block_text):
        line_full = block_text[ls:le]
        s = line_full[:-1] if line_full.endswith("\n") else line_full

        lines.append((ls, le, line_full, s))

        if _SEP_RE.match(s.strip()):
            segs_by_line.append([])
            continue

        segs = _line_segments_by_gaps(s, gap_min)
        segs = [g for g in segs if g.get("text", "").strip()]
        segs_by_line.append(segs)

    xs = []
    for segs in segs_by_line:
        for g in segs:
            txt = g["text"].strip()
            if len(txt) == 1 and txt in (":", "|", "-", "_"):
                continue
            xs.append(int(g["x"]))

    if not xs:
        txt = _strip_sep_lines(block_text)
        return [{
            "text": txt,
            "spans": [(abs_start, abs_start + len(block_text))],
            "start": abs_start,
            "end": abs_start + len(block_text),
            "layout_kind": "plain",
            "col_index": None,
            "block_start": abs_start,
            "block_end": abs_start + len(block_text),
        }]

    tol_cluster = 3 if is_ocr else 2
    centers = _cluster_centers(xs, tol=tol_cluster, min_hits=1)
    min_x = min(xs)
    if min_x not in centers:
        centers = sorted([min_x] + centers)
    centers = centers[:8]

    tol_assign = 6 if is_ocr else 4
    row_cells = []
    for (ls, le, line_full, s), segs in zip(lines, segs_by_line):
        r = {}
        for g in segs:
            ci = _assign_to_centers(int(g["x"]), centers, tol_assign)
            a = abs_start + ls + int(g["a"])
            b = abs_start + ls + int(g["b"])
            txt = g["text"].strip()

            if ci in r:
                t0, sp0 = r[ci]
                r[ci] = ((t0 + " " + txt).strip(), sp0 + [(a, b)])
            else:
                r[ci] = (txt, [(a, b)])
        row_cells.append(r)

    merge_dist = MERGE_COL_DIST_OCR if is_ocr else MERGE_COL_DIST_NATIVE
    centers, row_cells = _merge_close_columns(centers, row_cells, merge_dist=merge_dist)
    col_count = len(centers)

    if _is_micro_table_like(row_cells, col_count):
        table_rows = []
        for (ls, le, line_full, s) in lines:
            table_rows.append({"text": line_full, "spans": [(abs_start + ls, abs_start + le)]})

        table_cells = []
        for r in row_cells:
            row = []
            for ci in range(col_count):
                if ci in r:
                    t, sp = r[ci]
                    row.append({"col": ci, "text": t, "spans": [(int(a), int(b)) for a, b in sp if b > a]})
                else:
                    row.append({"col": ci, "text": "", "spans": []})
            table_cells.append(row)

        txt = _strip_sep_lines(block_text)
        return [{
            "text": txt,
            "spans": [(abs_start, abs_start + len(block_text))],
            "start": abs_start,
            "end": abs_start + len(block_text),
            "layout_kind": "header",
            "col_index": None,
            "block_start": abs_start,
            "block_end": abs_start + len(block_text),
            "table_rows": table_rows,
            "table_cells": table_cells,
            "header_source": "micro_multicol",
            "column_centers": centers,
        }]

    if _is_grid_like(row_cells, col_count):
        return [{
            "text": _strip_sep_lines(block_text),
            "spans": [(abs_start, abs_start + len(block_text))],
            "start": abs_start,
            "end": abs_start + len(block_text),
            "layout_kind": "multicol_grid",
            "col_index": None,
            "block_start": abs_start,
            "block_end": abs_start + len(block_text),
        }]

    col_items = []
    for ci in range(col_count):
        out_lines = []
        spans = []
        for r in row_cells:
            if ci in r:
                t, sp = r[ci]
                out_lines.append(t)
                spans.extend(sp)
            else:
                out_lines.append("")

        while out_lines and not out_lines[0].strip():
            out_lines.pop(0)
        while out_lines and not out_lines[-1].strip():
            out_lines.pop()

        compact = []
        blank = 0
        for ln in out_lines:
            if not ln.strip():
                blank += 1
                if blank <= 1:
                    compact.append("")
            else:
                blank = 0
                compact.append(ln)

        txt = "\n".join(compact).rstrip() + ("\n" if block_text.endswith("\n") else "")
        txt = _normalize_kv_generic(txt)

        if not _collapse_ws(txt).strip():
            continue

        if spans:
            st = min(a for a, _ in spans)
            en = max(b for _, b in spans)
        else:
            st = abs_start
            en = abs_start + len(block_text)

        col_items.append({
            "text": txt,
            "spans": [(int(a), int(b)) for (a, b) in spans if b > a],
            "start": st,
            "end": en,
            "layout_kind": "multicol_col",
            "col_index": ci,
            "block_start": abs_start,
            "block_end": abs_start + len(block_text),
        })

    if not col_items:
        return [{
            "text": _strip_sep_lines(block_text),
            "spans": [(abs_start, abs_start + len(block_text))],
            "start": abs_start,
            "end": abs_start + len(block_text),
            "layout_kind": "plain",
            "col_index": None,
            "block_start": abs_start,
            "block_end": abs_start + len(block_text),
        }]

    return col_items

def _looks_like_paragraphish(line_full: str) -> bool:
    s = (line_full or "").strip()
    if not s:
        return False
    if len(s) >= 120:
        words = re.findall(r"[A-Za-z---]+", s)
        if len(words) >= 10 and not _has_big_gap(s, 6, 1):
            return True
    return False

def _is_address_continuation_line(line_full: str, gap_min: int, is_ocr: bool) -> bool:
    s = (line_full or "").rstrip("\n")
    st = s.strip()
    if not st:
        return True
    if _SEP_RE.match(st):
        return True
    if _is_table_line(line_full, gap_min):
        return False
    if TABLE_HINT_RE.search(st):
        return False
    if _is_section_start_line(line_full):
        return False
    if _dec_tokens(st) > 0:
        return False
    if _num_tokens(st) > (4 if is_ocr else 6):
        return False
    if re.search(r"[A-Za-z---]", st) or _AR_RE.search(st):
        return True
    if re.match(r"^\d{4,6}$", st):
        return True
    if re.search(r"[@+/,-]", st) and len(st) <= 120:
        return True
    return False

def _collect_table_block(lines, start_i, gap_min):
    n = len(lines)
    i = start_i
    blank_run = 0
    seen_data = 0
    collected = []

    def _looks_like_wrap_line(s_raw: str) -> bool:
        if not s_raw:
            return False
        if not re.match(r"^[ \t]{2,}\S", s_raw):
            return False
        s_l = s_raw.lstrip(" \t")
        if _is_section_start_line(s_raw):
            return False
        if s_l.count("\t") >= 2:
            return False
        if _has_big_gap(s_l, gap_min, min_count=1):
            return False
        if _dec_tokens(s_l) != 0:
            return False
        if _num_tokens(s_l) > 1:
            return False
        return True

    while i < n:
        line_full, ls, le = lines[i]
        s = line_full.rstrip("\n")

        if not s.strip():
            blank_run += 1
            collected.append((line_full, ls, le))
            i += 1
            continue

        is_tbl = _is_table_line(line_full, gap_min)

        if is_tbl:
            blank_run = 0
            if _dec_tokens(s) >= 1 or _num_tokens(s) >= 3 or TABLE_HINT_RE.search(s):
                seen_data += 1
            collected.append((line_full, ls, le))
            i += 1
            continue

        if seen_data >= 1 and _looks_like_wrap_line(s):
            prev_nonblank = None
            for plf, _, _ in reversed(collected):
                if plf.strip():
                    prev_nonblank = plf.rstrip("\n")
                    break
            if prev_nonblank and (_is_table_line(prev_nonblank, gap_min) or _looks_like_wrap_line(prev_nonblank)):
                blank_run = 0
                collected.append((line_full, ls, le))
                i += 1
                continue

        if seen_data >= 2 and blank_run >= 2:
            break
        if seen_data >= 1 and blank_run >= 1:
            break
        break

    while collected and not collected[-1][0].strip():
        collected.pop()

    return collected, i

def _make_span_item(page_text, spans, text_override, kind, meta=None):
    spans2 = [(int(a), int(b)) for (a, b) in (spans or []) if b > a]
    if spans2:
        st = min(a for a, _ in spans2)
        en = max(b for _, b in spans2)
    else:
        st = 0
        en = 0
    it = {"text": text_override, "spans": spans2, "start": st, "end": en, "layout_kind": kind}
    if meta:
        it.update(meta)
    return it

def layout_items(page_text: str, lang: str, extraction: str = ""):
    if not page_text:
        return []

    is_ocr = str(extraction or "").startswith("ocr:")
    gap_min = GAP_MIN_OCR if is_ocr else GAP_MIN_NATIVE

    lines = []
    for ls, le in _iter_line_spans(page_text):
        lines.append((page_text[ls:le], ls, le))

    items = []
    i = 0
    n = len(lines)

    def _starts_table(i0):
        return _is_table_line(lines[i0][0], gap_min)

    def _starts_multicol(i0):
        return _is_multicol_candidate_line(lines[i0][0], gap_min=gap_min, is_ocr=is_ocr)

    while i < n:
        if _starts_table(i):
            collected, j = _collect_table_block(lines, i, gap_min=gap_min)
            if collected:
                a0 = collected[0][1]
                b0 = collected[-1][2]
                block_text = page_text[a0:b0]
                table_rows = [{"text": lf, "spans": [(lls, lle)]} for (lf, lls, lle) in collected]
                items.append(_make_span_item(
                    page_text,
                    spans=[(a0, b0)],
                    text_override=block_text,
                    kind="table",
                    meta={"table_rows": table_rows}
                ))
                i = j
                continue

        if _starts_multicol(i):
            start = i

            if start - 1 >= 0:
                prev_line = lines[start - 1][0]
                if _looks_like_title_line(prev_line) and not _starts_table(start - 1):
                    start -= 1

            j = i
            saw_any = False
            blank_run = 0
            noncol_inside = 0

            MAX_INBLOCK_BLANK = 6
            MAX_INBLOCK_LINES = 140
            MAX_NONCOL_INSIDE = 25
            weak_gap = max(3, gap_min - (3 if is_ocr else 2))

            while j < n and (j - start) < MAX_INBLOCK_LINES:
                if _starts_table(j):
                    break

                lf, lls, lle = lines[j]
                ss = lf.rstrip("\n")

                if not ss.strip() or _SEP_RE.match(ss.strip()):
                    blank_run += 1
                    j += 1
                    if saw_any and blank_run >= MAX_INBLOCK_BLANK:
                        break
                    continue

                blank_run = 0

                if _starts_multicol(j):
                    saw_any = True
                    noncol_inside = 0
                    j += 1
                    continue

                if saw_any and noncol_inside < MAX_NONCOL_INSIDE:
                    if _is_address_continuation_line(lf, gap_min=gap_min, is_ocr=is_ocr) and not _looks_like_paragraphish(lf):
                        noncol_inside += 1
                        j += 1
                        continue
                    if _has_big_gap(ss, weak_gap, min_count=1) and not _looks_like_paragraphish(lf):
                        noncol_inside += 1
                        j += 1
                        continue

                break

            end = j if j > i else i + 1

            a0 = lines[start][1]
            b0 = lines[end-1][2] if end-1 >= start else lines[start][2]
            block_text = page_text[a0:b0]

            items.extend(_transpose_or_group_multicol(block_text, abs_start=a0, gap_min=gap_min, is_ocr=is_ocr))

            i = end
            continue

        start = i
        j = i
        while j < n:
            if _starts_table(j) or _starts_multicol(j):
                break
            j += 1

        a0 = lines[start][1]
        b0 = lines[j-1][2] if j-1 >= start else lines[start][2]
        plain_text = page_text[a0:b0]

        chunks = chunk_layout_universal(plain_text, lang)
        pos = 0
        for ch in chunks:
            ca = a0 + pos
            cb = ca + len(ch)
            pos += len(ch)
            items.append(_make_span_item(page_text, spans=[(ca, cb)], text_override=ch, kind="plain"))

        i = j if j > start else i + 1

    def _k(it):
        if it.get("layout_kind") in ("multicol_col", "multicol_grid"):
            return (it.get("block_start", it.get("start", 0)),
                    it.get("col_index", 0) if it.get("col_index") is not None else -1)
        return (it.get("start", 0), 0)

    items.sort(key=_k)
    return items

# ==================== Noise detection (audit) ====================
_NOISE_LINE_RE = re.compile(
    r"(?i)^\s*(sample|confidential|draft)\s*$|"
    r"^\s*page\s+\d+\s*(?:of|/)\s*\d+\s*$|"
    r"^\s*\d+\s*(?:of|/)\s*\d+\s*$"
)

def build_noise_keys_for_doc(pages_text):
    if not pages_text:
        return set()
    page_count = len(pages_text)
    if page_count < 3:
        return set()
    min_pages = max(3, int(math.ceil(page_count * 0.30)))

    counts = {}
    counts_masked = {}

    for txt in pages_text:
        seen = set()
        seen_m = set()
        for ls, le in _iter_line_spans(txt or ""):
            line = (txt[ls:le]).rstrip("\n")
            key = _collapse_ws(line).lower()
            if not key:
                continue

            if _SEP_RE.match(key) or _NOISE_LINE_RE.match(line):
                counts[key] = counts.get(key, 0) + 1
                continue

            mkey = _mask_digits(key)

            if key not in seen:
                counts[key] = counts.get(key, 0) + 1
                seen.add(key)
            if mkey not in seen_m:
                counts_masked[mkey] = counts_masked.get(mkey, 0) + 1
                seen_m.add(mkey)

    noise_keys = set()
    for k, c in counts.items():
        if c >= min_pages:
            noise_keys.add(k)
    for mk, c in counts_masked.items():
        if c >= min_pages:
            noise_keys.add(mk)

    return noise_keys

def chunk_is_noise(chunk_text: str, noise_keys: set) -> bool:
    if not chunk_text:
        return True

    has_nonempty = False
    for ls, le in _iter_line_spans(chunk_text):
        line = chunk_text[ls:le].rstrip("\n")
        st = line.strip()
        if not st:
            continue
        if _SEP_RE.match(st):
            continue

        has_nonempty = True
        key = _collapse_ws(line).lower()
        mkey = _mask_digits(key)

        if _NOISE_LINE_RE.match(line):
            continue
        if key in noise_keys or mkey in noise_keys:
            continue

        return False

    return True if has_nonempty else True

# ==================== Helpers emplacement (page) ====================
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

def _nonspace_len(s: str) -> int:
    return len(_WS_RE.sub("", s or ""))

def _line_col_from_offset(text: str, off: int):
    if off < 0:
        off = 0
    if off > len(text):
        off = len(text)
    line = text.count("\n", 0, off) + 1
    last_nl = text.rfind("\n", 0, off)
    col = off if last_nl < 0 else (off - last_nl - 1)
    return line, col

# ==================== Metadonnes depuis DOCS / TEXT_DOCS ====================
def _safe_str(x):
    try:
        return str(x)
    except Exception:
        return ""

def _unique_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _pdf_extract_pages_text(path: str):
    PdfReader = _get_pdf_reader()
    if PdfReader is None:
        return None
    try:
        reader = PdfReader(path)
        out = []
        for p in reader.pages:
            out.append(p.extract_text() or "")
        return out
    except Exception:
        return None

def _pdf_page_count(path: str):
    PdfReader = _get_pdf_reader()
    if PdfReader is None:
        return None
    try:
        return len(PdfReader(path).pages)
    except Exception:
        return None

# ==================== Vrifier FINAL_DOCS (mme logique que test) ====================
if "FINAL_DOCS" not in globals() or not isinstance(FINAL_DOCS, list):
    raise RuntimeError("FINAL_DOCS not found. Excute d'abord la cellule prcdente (celle qui imprime FINAL PRINT).")

# ==================== Construire une structure DOC -> PAGES ====================
DOC_PACK = []

# 1) OCR: DOCS (si dispo)
if "DOCS" in globals() and isinstance(DOCS, list):
    for d in DOCS:
        doc_id = d.get("doc_id")
        filename = d.get("filename") or "unknown"
        pages = d.get("pages", []) or []
        page_count_total = d.get("page_count_total") if d.get("page_count_total") else len(pages)

        paths = []
        for p in pages:
            sp = p.get("source_path") or p.get("path")
            if sp:
                paths.append(_safe_str(sp))
        paths = _unique_keep_order(paths)

        pages_out = []
        for p in pages:
            pages_out.append({
                "page_index": int(p.get("page_index") or 1),
                "text": p.get("ocr_text") or "",
                "source_path": _safe_str(p.get("source_path") or p.get("path") or ""),
            })
        pages_out.sort(key=lambda x: x["page_index"])

        DOC_PACK.append({
            "doc_id": doc_id,
            "filename": filename,
            "content": "image_only",
            "extraction": "ocr:tesseract",
            "paths": paths,
            "page_count_total": page_count_total,
            "pages": pages_out,
        })

# 2) NATIVE: TEXT_DOCS (si dispo)
if "TEXT_DOCS" in globals() and isinstance(TEXT_DOCS, list):
    for d in TEXT_DOCS:
        doc_id = d.get("doc_id")
        filename = d.get("filename") or "unknown"
        extraction = d.get("extraction") or "native:unknown"
        sp = d.get("source_path") or ""
        paths = _unique_keep_order([_safe_str(sp)]) if sp else []
        full_text = d.get("text") or ""

        pages_out = []
        page_count_total = d.get("page_count_total", None)
        pages_text = d.get("pages_text", None)

        if pages_text is not None and isinstance(pages_text, list) and len(pages_text) > 0:
            page_count_total = page_count_total or len(pages_text)
            for i2, txt in enumerate(pages_text, start=1):
                pages_out.append({
                    "page_index": i2,
                    "text": txt or "",
                    "source_path": _safe_str(sp),
                })
        else:
            if sp and str(sp).lower().endswith(".pdf") and Path(sp).exists():
                pages_text2 = _pdf_extract_pages_text(sp)
                if pages_text2:
                    page_count_total = page_count_total or len(pages_text2)
                    for i2, txt in enumerate(pages_text2, start=1):
                        pages_out.append({
                            "page_index": i2,
                            "text": txt or "",
                            "source_path": _safe_str(sp),
                        })
                else:
                    pages_out.append({
                        "page_index": 1,
                        "text": full_text,
                        "source_path": _safe_str(sp),
                    })
                    page_count_total = page_count_total or 1
            else:
                pages_out.append({
                    "page_index": 1,
                    "text": full_text,
                    "source_path": _safe_str(sp),
                })
                page_count_total = page_count_total or 1

        if page_count_total is None and sp and str(sp).lower().endswith(".pdf") and Path(sp).exists():
            pc = _pdf_page_count(sp)
            if pc is not None:
                page_count_total = pc

        DOC_PACK.append({
            "doc_id": doc_id,
            "filename": filename,
            "content": "text",
            "extraction": extraction,
            "paths": paths,
            "page_count_total": page_count_total,
            "pages": pages_out,
        })

# 3) Fallback  FINAL_DOCS
if not DOC_PACK:
    for d in FINAL_DOCS:
        DOC_PACK.append({
            "doc_id": d.get("doc_id"),
            "filename": d.get("filename") or "unknown",
            "content": d.get("content"),
            "extraction": d.get("extraction"),
            "paths": [],
            "page_count_total": 1,
            "pages": [{"page_index": 1, "text": d.get("text") or "", "source_path": ""}],
        })

# ==================== Tokeniser: construire TOK_DOCS ====================
TOK_DOCS = []

for doc in DOC_PACK:
    doc_id = doc.get("doc_id")
    filename = doc.get("filename") or "unknown"
    extraction = doc.get("extraction") or ""
    content_type = doc.get("content")
    paths = doc.get("paths") or []
    page_count_total = doc.get("page_count_total")

    pages_text_for_noise = [(p.get("text") or "") for p in (doc.get("pages") or [])]
    noise_keys = build_noise_keys_for_doc(pages_text_for_noise)

    pages_tok = []
    doc_chars_total = 0
    recompose_ok_doc = True

    for pg in (doc.get("pages") or []):
        page_index = int(pg.get("page_index") or 1)
        page_text = pg.get("text") or ""
        doc_chars_total += len(page_text)

        lang = detect_lang(page_text)

        items = layout_items(page_text, lang, extraction=extraction)
        recompose_ok = False if any(it.get("layout_kind") in ("multicol_col", "multicol_grid", "table", "header") for it in items) else True
        if not recompose_ok:
            recompose_ok_doc = False

        sent_items = []
        for it in items:
            chunk = it["text"]
            start = int(it.get("start", 0))
            end = int(it.get("end", start + len(chunk)))

            line, col = _line_col_from_offset(page_text, start)
            nonspace = _nonspace_len(chunk)

            is_noise = chunk_is_noise(chunk, noise_keys)

            if it.get("layout_kind") in ("multicol_col", "multicol_grid", "table", "header"):
                is_sentence = (not is_noise) and (nonspace >= 1)
            else:
                is_sentence = (nonspace >= MIN_SENTENCE_NONSPACE) and (not is_noise)

            sent_items.append({
                "text": chunk,
                "start": start,
                "end": end,
                "line": line,
                "col": col,
                "chars": len(chunk),
                "nonspace": nonspace,
                "is_noise": is_noise,
                "is_sentence": is_sentence,
                "spans": it.get("spans", []),
                "layout_kind": it.get("layout_kind", "plain"),
                "col_index": it.get("col_index", None),
                "table_rows": it.get("table_rows", None),
                # IMPORTANT: mme champs que test (pas de renommage)
                "header_rows": it.get("table_rows", None),
                "header_cells": it.get("table_cells", None),
                "header_source": it.get("header_source", None),
            })

        pages_tok.append({
            "page_index": page_index,
            "source_path": pg.get("source_path") or "",
            "lang": lang,
            "chars": len(page_text),
            "recompose_ok": recompose_ok,
            "sentences_layout": sent_items,
            "page_text": page_text,
        })

    pages_tok.sort(key=lambda x: x["page_index"])

    TOK_DOCS.append({
        "doc_id": doc_id,
        "filename": filename,
        "paths": paths,
        "page_count_total": page_count_total,
        "content": content_type,
        "extraction": extraction,
        "pages": pages_tok,
        "chars_total": doc_chars_total,
        "recompose_ok": recompose_ok_doc,
    })

def _sort_key(x):
    p = (x.get("paths") or [""])[0]
    return (x.get("filename") or "", str(p))

TOK_DOCS.sort(key=_sort_key)

TOK_BY_ID = {d["doc_id"]: d for d in TOK_DOCS if d.get("doc_id")}
TOK_BY_FILENAME = {}
for d in TOK_DOCS:
    TOK_BY_FILENAME.setdefault(d["filename"], []).append(d)

def _select_doc(target):
    if target is None:
        return TOK_DOCS
    if isinstance(target, int):
        if 0 <= target < len(TOK_DOCS):
            return [TOK_DOCS[target]]
        raise IndexError(f"TARGET index out of range: {target} (0..{len(TOK_DOCS)-1})")
    if isinstance(target, str):
        t = target.strip()
        if t in TOK_BY_ID:
            return [TOK_BY_ID[t]]
        if t in TOK_BY_FILENAME:
            return TOK_BY_FILENAME[t]
        hits = []
        for d in TOK_DOCS:
            if t.lower() in (d.get("filename","").lower()):
                hits.append(d)
                continue
            for p in d.get("paths") or []:
                if t.lower() in str(p).lower():
                    hits.append(d)
                    break
        if hits:
            return hits
        raise ValueError(f"No document matches TARGET='{target}' (by doc_id/filename/path).")
    raise TypeError("TARGET must be None, int, or str")

def print_one_doc(doc):
    print("=" * 120)
    print(f"[doc] {doc['filename']}")
    print(f"  doc_id       : {doc.get('doc_id')}")
    print(f"  content      : {doc.get('content')}")
    print(f"  extraction   : {doc.get('extraction')}")
    print(f"  pages_total  : {doc.get('page_count_total')}")
    print(f"  chars_total  : {doc.get('chars_total')}")
    print(f"  recompose_ok : {doc.get('recompose_ok')}")
    print("  paths:")
    if doc.get("paths"):
        for p in doc["paths"]:
            print(f"    - {p}")
    else:
        print("    - (unknown)")
    print("-" * 120)

    if not PRINT_SENTENCES:
        return

    for pg in (doc.get("pages") or []):
        print(f"[page {pg['page_index']}/{doc.get('page_count_total') or '?'}] source_path={pg.get('source_path')}")
        print(f"  lang         : {pg.get('lang')}")
        print(f"  chars        : {pg.get('chars')}")
        print("-" * 120)

        if PRINT_PAGE_TEXT:
            # IMPORTANT: prserver exactement la page (viter strip)
            txt = pg.get("page_text") or ""
            print(txt, end="" if txt.endswith("\n") else "\n")
            print("-" * 120)

        sent_items = pg.get("sentences_layout") or []

        total_all = len(sent_items)
        total_noise = sum(1 for s in sent_items if s.get("is_noise"))
        total_sentence = sum(1 for s in sent_items if s.get("is_sentence"))

        if PRINT_ONLY_SENTENCES:
            view = [s for s in sent_items if s.get("is_sentence")]
        else:
            view = list(sent_items)

        fallback_used = False
        if PRINT_ONLY_SENTENCES and not view and total_all > 0:
            view = list(sent_items)
            fallback_used = True

        total_view = len(view)
        show = total_view if MAX_SENTENCES_PREVIEW is None else min(total_view, MAX_SENTENCES_PREVIEW)

        print(
            f"  sentences_layout: {total_all} chunks total | "
            f"sentences={total_sentence} | noise={total_noise} | "
            f"showing {show}/{total_view} "
            f"(filter_is_sentence={PRINT_ONLY_SENTENCES}, fallback={fallback_used}, min_nonspace={MIN_SENTENCE_NONSPACE})"
        )
        print("-" * 120)

        for i2 in range(show):
            s = view[i2]
            chunk = s["text"]
            print(
                f"[sent {i2+1}/{total_view}] page={pg['page_index']} start={s['start']} end={s['end']} "
                f"line={s['line']} col={s['col']} chars={s['chars']} nonspace={s['nonspace']} "
                f"is_noise={s.get('is_noise')} is_sentence={s['is_sentence']} layout={s.get('layout_kind')}"
            )
            # IMPORTANT: sortie identique au test (respect \n / pas d'ajout)
            print(chunk, end="" if chunk.endswith("\n") else "\n")
            if PRINT_REPR:
                print("repr:", repr(chunk))
            print("-" * 80)

        if MAX_SENTENCES_PREVIEW is not None and total_view > show:
            print(f"... {total_view - show} chunks restants non affichs (MAX_SENTENCES_PREVIEW={MAX_SENTENCES_PREVIEW})")

        print()

# ==================== Excution ====================
selected = _select_doc(TARGET)

if not selected:
    print("[info] Aucun document  traiter.")
else:
    for doc in selected:
        print_one_doc(doc)






