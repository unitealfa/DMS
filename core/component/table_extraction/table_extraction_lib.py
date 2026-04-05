from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pipeline.file_resolution import resolve_runtime_input_path
except Exception:
    def resolve_runtime_input_path(path: Path, repo_root: Path) -> Path:
        return path


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

FIELD_SYNONYMS: Dict[str, List[str]] = {
    "reference": [
        "reference",
        "ref",
        "ref.",
        "code",
        "code article",
        "product code",
        "sku",
        "item code",
    ],
    "product": [
        "product",
        "produit",
        "article",
        "designation",
        "description",
        "item",
        "items",
        "libelle",
        "service",
        "services",
    ],
    "quantity": [
        "qty",
        "qte",
        "qt",
        "qnty",
        "quantite",
        "quantite",
        "quantity",
        "units",
        "unites",
        "nombre",
        "nb",
    ],
    "unit_price": [
        "prix unitaire",
        "prix unit",
        "pu",
        "unit price",
        "price unit",
        "price",
        "tarif unitaire",
        "rate",
    ],
    "total_ht": [
        "total ht",
        "montant ht",
        "hors taxe",
        "net ht",
    ],
    "total_ttc": [
        "total ttc",
        "montant ttc",
        "toutes taxes comprises",
        "incl tax",
        "gross total",
        "total taxes",
    ],
    "total": [
        "total",
        "montant",
        "amount",
        "valeur",
        "line total",
        "net",
    ],
    "tax": [
        "tva",
        "vat",
        "tax",
        "taxe",
    ],
}

TABLE_HINT_TERMS = {
    "QTE",
    "QTY",
    "QNTY",
    "QUANTITE",
    "QUANTITY",
    "PRIX",
    "PRICE",
    "UNIT",
    "UNITAIRE",
    "UNITAIRE",
    "MONTANT",
    "AMOUNT",
    "TOTAL",
    "HT",
    "TTC",
    "TVA",
    "VAT",
    "PRODUCT",
    "PRODUIT",
    "ARTICLE",
    "DESIGNATION",
    "DESCRIPTION",
    "REFERENCE",
    "REF",
    "ITEM",
    "ITEMS",
    "SERVICE",
    "SERVICES",
    "VALEUR",
    "P.UNITAIRE",
    "P UNITAIRE",
}

HEADER_STRONG_HINTS = {
    "REFERENCE",
    "REF",
    "PRODUCT",
    "PRODUIT",
    "ARTICLE",
    "DESIGNATION",
    "DESCRIPTION",
    "ITEM",
    "ITEMS",
    "SERVICE",
    "SERVICES",
    "QTE",
    "QTY",
    "QNTY",
    "QUANTITE",
    "QUANTITY",
    "PRIX",
    "PRICE",
    "UNIT",
    "UNITAIRE",
    "P.UNITAIRE",
    "P UNITAIRE",
    "VALEUR",
}

HEADER_WEAK_HINTS = {
    "TOTAL",
    "AMOUNT",
    "MONTANT",
    "HT",
    "TTC",
    "TVA",
    "VAT",
}

TOTALS_STOP_HINTS = {
    "SUBTOTAL",
    "SOUS TOTAL",
    "TOTAL TTC",
    "TOTAL HT",
    "TOTAL DUE",
    "AMOUNT DUE",
    "MONTANT A PAYER",
    "MONTANT TTC",
    "MONTANT HT",
    "TAX",
    "TAXES",
    "TVA",
    "VAT",
    "TIMBRE",
    "STAMP",
    "NET A PAYER",
    "NET PAYABLE",
    "REMISE",
    "DISCOUNT",
    "FRAIS",
    "FRAIS DE PORT",
    "FRAIS DE LIVRAISON",
    "LIVRAISON",
    "SHIPPING",
    "SHIPPING COST",
    "PORT",
    "ACOMPTE",
    "ADVANCE",
}

FOOTER_ONLY_HINTS = {
    "NON ASSUJETTI",
    "NON ASSUJETTI A LA TVA",
    "MODE DE PAIEMENT",
    "PAYMENT METHOD",
    "DATE ECHEANCE",
    "ECHEANCE",
    "SIGNATURE",
    "CACHET",
    "FACTURE EN LETTRE",
    "TERMS CONDITIONS",
    "DATA INFO",
    "AMOUNT DUE",
}

NON_TABLE_LEFT_HINTS = {
    "TEL",
    "TELEPHONE",
    "PHONE",
    "DATE",
    "IBAN",
    "SWIFT",
    "BIC",
    "FISCAL",
    "IDENT",
    "CLIENT",
    "ADRESSE",
    "ADDRESS",
    "PAIEMENT",
    "PAYMENT",
    "ECHEANCE",
    "EFFECTIVE",
    "SIGNATURE",
    "CACHET",
    "N ART",
    "N°ART",
}

TOTAL_FIELD_HINTS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("TOTAL TTC", "MONTANT A PAYER TTC", "MONTANT A PAYER TTE", "MONTANT TTC", "TOTAL A PAYER TTC"), "total_ttc"),
    (("TOTAL HT", "MONTANT HT", "TOTAL H.T", "SUBTOTAL", "SOUS TOTAL"), "total_ht"),
    (("MONTANT A PAYER", "TOTAL A PAYER", "AMOUNT DUE", "TOTAL DUE"), "amount_due"),
    (("TIMBRE", "STAMP"), "stamp"),
    (("TVA", "VAT", "TAX", "TAXES"), "tax"),
    (("TOTAL", "MONTANT", "AMOUNT"), "total"),
)

LARGE_GAP_MIN = 3
ANCHOR_TOL = 3


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _doc_key(doc: Dict[str, Any], idx: int) -> str:
    sid = str(doc.get("doc_id") or "").strip()
    if sid:
        return f"id:{sid}"
    sfn = str(doc.get("filename") or "").strip().lower()
    if sfn:
        return f"fn:{sfn}"
    return f"idx:{idx}"


def _doc_text_score(doc: Dict[str, Any]) -> int:
    score = 0
    for page in _safe_list(doc.get("pages")):
        if not isinstance(page, dict):
            continue
        score += len(_compact_spaces(page.get("page_text") or page.get("ocr_text") or page.get("text") or ""))
    score += len(_compact_spaces(doc.get("text") or ""))
    return score


def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        key = _doc_key(doc, i)
        if key not in best:
            best[key] = doc
            order.append(key)
            continue
        if _doc_text_score(doc) > _doc_text_score(best[key]):
            best[key] = doc
    return [best[k] for k in order]


def _norm_text(value: Any) -> str:
    txt = str(value or "")
    txt = txt.translate(ARABIC_DIGITS)
    txt = txt.replace("\xa0", " ").replace("’", "'").replace("`", "'")
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"[^A-Za-z0-9+./%-]+", " ", txt)
    return " ".join(txt.upper().split())


def _compact_spaces(value: Any) -> str:
    txt = str(value or "")
    txt = txt.replace("\xa0", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(txt.split())


def _line_token_spans(line: Any) -> List[Tuple[str, int, int]]:
    raw = str(line or "").replace("\xa0", " ").rstrip("\r\n")
    out: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", raw):
        token = str(m.group(0) or "")
        if not token:
            continue
        out.append((token, int(m.start()), int(m.end()) - 1))
    return out


def _line_segment_spans(line: Any, min_gap: int = LARGE_GAP_MIN) -> List[Dict[str, Any]]:
    spans = _line_token_spans(line)
    if not spans:
        return []

    groups: List[List[Tuple[str, int, int]]] = [[spans[0]]]
    for token, start, end in spans[1:]:
        prev_end = groups[-1][-1][2]
        gap = int(start) - int(prev_end) - 1
        if gap >= int(min_gap):
            groups.append([(token, start, end)])
        else:
            groups[-1].append((token, start, end))

    out: List[Dict[str, Any]] = []
    for group in groups:
        toks = [g[0] for g in group]
        start = int(group[0][1])
        end = int(group[-1][2])
        out.append(
            {
                "text": " ".join(toks).strip(),
                "start": start,
                "end": end,
                "tokens": toks,
            }
        )
    return out


def _line_geometry(line: Any) -> Dict[str, Any]:
    raw = str(line or "").replace("\xa0", " ").rstrip("\r\n")
    token_spans = _line_token_spans(raw)
    segments = _line_segment_spans(raw)
    starts = [int(s[1]) for s in token_spans]
    ends = [int(s[2]) for s in token_spans]
    tokens = [str(s[0]) for s in token_spans]

    gaps: List[int] = []
    for i in range(1, len(token_spans)):
        gap = starts[i] - ends[i - 1] - 1
        gaps.append(max(0, int(gap)))
    large_gaps = [g for g in gaps if g >= LARGE_GAP_MIN]

    alpha_tokens = sum(1 for tok in tokens if any(ch.isalpha() for ch in tok))
    numeric_tokens = sum(1 for tok in tokens if _to_amount(tok) is not None or _to_quantity(tok) is not None)
    punct_tokens = sum(1 for tok in tokens if not any(ch.isalnum() for ch in tok))
    n_tokens = len(tokens)
    alpha_ratio = (float(alpha_tokens) / float(max(1, n_tokens))) if n_tokens else 0.0
    digit_ratio = (float(numeric_tokens) / float(max(1, n_tokens))) if n_tokens else 0.0
    punct_ratio = (float(punct_tokens) / float(max(1, n_tokens))) if n_tokens else 0.0
    left_indent = len(raw) - len(raw.lstrip(" "))
    rightmost = max(ends) if ends else -1

    cells = _split_line_cells(raw)
    header_like = _is_header_like_line(raw, cells)
    paragraph_like = float(n_tokens >= 8 and len(large_gaps) == 0 and alpha_ratio >= 0.65)
    mixed_type_like = float(alpha_tokens >= 1 and numeric_tokens >= 1 and len(large_gaps) >= 1)

    return {
        "raw": raw,
        "tokens": tokens,
        "token_starts": starts,
        "token_ends": ends,
        "segments": segments,
        "n_tokens": n_tokens,
        "gap_sizes": gaps,
        "large_gap_count": len(large_gaps),
        "max_gap": max(large_gaps) if large_gaps else 0,
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
        "punctuation_ratio": punct_ratio,
        "left_indent": left_indent,
        "rightmost_char": rightmost,
        "cells": cells,
        "header_like": header_like,
        "paragraph_like": paragraph_like,
        "mixed_type_like": mixed_type_like,
    }


def _alignment_with_neighbor(cur: Dict[str, Any], other: Dict[str, Any], tol: int = ANCHOR_TOL) -> float:
    cur_starts = [int(v) for v in _safe_list(cur.get("token_starts"))]
    other_starts = [int(v) for v in _safe_list(other.get("token_starts"))]
    if not cur_starts or not other_starts:
        return 0.0
    matched = 0
    for pos in cur_starts:
        if any(abs(pos - p2) <= int(tol) for p2 in other_starts):
            matched += 1
    return float(matched) / float(max(1, len(cur_starts)))


def _line_tabularity_score(geoms: List[Dict[str, Any]], idx: int) -> float:
    geom = geoms[idx]
    n_tokens = int(geom.get("n_tokens") or 0)
    if n_tokens <= 0:
        return -1.0
    prev_align = _alignment_with_neighbor(geom, geoms[idx - 1]) if idx > 0 else 0.0
    next_align = _alignment_with_neighbor(geom, geoms[idx + 1]) if idx + 1 < len(geoms) else 0.0

    score = 0.0
    score += 0.80 * float(min(4, int(geom.get("large_gap_count") or 0)))
    score += 0.20 * float(min(8, n_tokens))
    score += 1.40 * float(max(prev_align, next_align))
    score += 1.90 if bool(geom.get("header_like")) else 0.0
    score += 0.85 * float(geom.get("mixed_type_like") or 0.0)
    score -= 1.30 * float(geom.get("paragraph_like") or 0.0)
    if n_tokens <= 1 and not bool(geom.get("header_like")):
        score -= 0.50
    return score


def _anchors_from_header_geom(header_geom: Dict[str, Any]) -> List[int]:
    anchors: List[int] = []
    for seg in _safe_list(header_geom.get("segments")):
        if not isinstance(seg, dict):
            continue
        start = int(seg.get("start") or 0)
        if not anchors or abs(start - anchors[-1]) > 1:
            anchors.append(start)
    if len(anchors) >= 2:
        return anchors

    starts = [int(v) for v in _safe_list(header_geom.get("token_starts"))]
    for start in starts:
        if not anchors or abs(start - anchors[-1]) > LARGE_GAP_MIN:
            anchors.append(start)
    return anchors


def _anchor_matches(segments: List[Dict[str, Any]], anchors: List[int], tol: int = ANCHOR_TOL) -> Tuple[int, float]:
    if not segments or not anchors:
        return 0, 0.0
    matches = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = int(seg.get("start") or 0)
        if any(abs(start - a) <= int(tol) for a in anchors):
            matches += 1
    return matches, float(matches) / float(max(1, len(segments)))


def _cells_from_anchors(segments: List[Dict[str, Any]], anchors: List[int]) -> List[str]:
    if not anchors:
        return []
    cells = [""] * len(anchors)
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = _compact_spaces(seg.get("text"))
        if not text:
            continue
        start = int(seg.get("start") or 0)
        col_idx = min(range(len(anchors)), key=lambda i: abs(start - anchors[i]))
        if cells[col_idx]:
            cells[col_idx] = f"{cells[col_idx]} {text}".strip()
        else:
            cells[col_idx] = text
    while cells and not _compact_spaces(cells[-1]):
        cells.pop()
    return cells


def _collect_blocks_anchor(lines: List[str]) -> List[List[Dict[str, Any]]]:
    geoms = [_line_geometry(line) for line in lines]
    if not geoms:
        return []

    for i in range(len(geoms)):
        geoms[i]["tabular_score"] = _line_tabularity_score(geoms, i)

    candidate_idxs: List[int] = []
    for i, geom in enumerate(geoms):
        segments_count = len(_safe_list(geom.get("segments")))
        if segments_count < 2:
            continue
        header_like = bool(geom.get("header_like"))
        score = float(geom.get("tabular_score") or 0.0)
        if header_like and segments_count >= 2:
            candidate_idxs.append(i)
            continue
        if score >= 2.6 and int(geom.get("large_gap_count") or 0) >= 1 and segments_count >= 3:
            candidate_idxs.append(i)

    blocks: List[List[Dict[str, Any]]] = []
    occupied_until = -1
    for header_idx in candidate_idxs:
        if header_idx <= occupied_until:
            continue
        header_geom = geoms[header_idx]
        anchors = _anchors_from_header_geom(header_geom)
        if len(anchors) < 2:
            continue

        header_cells = _cells_from_anchors(_safe_list(header_geom.get("segments")), anchors)
        if len(header_cells) < 2:
            continue
        if not _is_header_like_line(str(header_geom.get("raw") or ""), header_cells):
            if int(header_geom.get("large_gap_count") or 0) < 2:
                continue

        block_rows: List[Dict[str, Any]] = [{"line": str(header_geom.get("raw") or ""), "cells": header_cells}]
        miss_count = 0
        end_idx = header_idx
        anchor_aligned_rows = 1

        for j in range(header_idx + 1, len(geoms)):
            geom = geoms[j]
            raw = str(geom.get("raw") or "")
            segments = _safe_list(geom.get("segments"))
            if not _compact_spaces(raw):
                miss_count += 1
                if miss_count > 1:
                    break
                continue

            match_count, match_ratio = _anchor_matches(segments, anchors)
            cells = _cells_from_anchors(segments, anchors)
            if len(cells) < 2:
                cells = _split_line_cells(raw)
            line_tabular = _line_looks_tabular(cells, raw)
            footer_like = _is_footer_like_line(raw, cells)
            hard_stop = (
                footer_like
                or (
                    _line_header_hint_score(raw) == 0
                    and any(h in _norm_text(raw) for h in NON_TABLE_LEFT_HINTS)
                    and match_count == 0
                    and not line_tabular
                )
            )
            if hard_stop:
                break

            compatible = bool(
                (not footer_like)
                and (
                    (match_ratio >= 0.50)
                    or (match_count >= 2)
                    or line_tabular
                    or (match_count >= 1 and _is_probable_code(str(cells[0] if cells else "")))
                )
            )

            if compatible:
                block_rows.append({"line": raw, "cells": cells})
                end_idx = j
                miss_count = 0
                if match_ratio >= 0.50 or match_count >= 2:
                    anchor_aligned_rows += 1
            else:
                miss_count += 1
                if miss_count >= 2:
                    break

        if len(block_rows) < 2:
            continue
        if anchor_aligned_rows < 2 and len(block_rows) < 3:
            continue
        blocks.append(block_rows)
        occupied_until = max(occupied_until, end_idx)

    return blocks


def _merge_blocks(primary: List[List[Dict[str, Any]]], secondary: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    seen = set()
    for block in list(primary) + list(secondary):
        if not isinstance(block, list) or not block:
            continue
        line_keys = [_compact_spaces(row.get("line")) for row in block if isinstance(row, dict)]
        line_keys = [v for v in line_keys if v]
        if len(line_keys) < 2:
            continue
        sig = (line_keys[0], line_keys[-1], len(line_keys))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(block)
    return out


def _split_line_cells(line: str) -> List[str]:
    raw = str(line or "").replace("\xa0", " ").replace("\r", "")
    if not raw:
        return []

    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
    elif "\t" in str(line or ""):
        parts = [p.strip() for p in str(line).split("\t")]
    else:
        parts = [p.strip() for p in re.split(r"\s{2,}", raw.strip())]
        if len(parts) < 2 and ";" in raw:
            parts = [p.strip() for p in raw.split(";")]

    parts = [p for p in parts if p]
    if len(parts) >= 2:
        return parts

    dense = _split_dense_numeric_tail(raw)
    if len(dense) >= 2:
        return dense

    header_dense = _split_dense_header_cells(raw)
    if len(header_dense) >= 2:
        return header_dense
    return parts


def _is_probable_code(value: str) -> bool:
    txt = _compact_spaces(value).translate(ARABIC_DIGITS)
    if not txt or " " in txt:
        return False
    if not any(ch.isalpha() for ch in txt):
        return False
    if not any(ch.isdigit() for ch in txt):
        return False
    return bool(re.fullmatch(r"[A-Za-z]{0,6}[-_/]?[A-Za-z0-9]{2,}", txt))


def _normalize_reference_code(value: Any) -> Optional[str]:
    txt = _compact_spaces(value).translate(ARABIC_DIGITS).upper()
    if not txt:
        return None
    txt = re.sub(r"[^A-Z0-9_-]", "", txt)
    if not txt:
        return None
    if re.fullmatch(r"\d{3,8}", txt):
        return txt
    if not _is_probable_code(txt):
        return None
    return txt


def _extract_reference_code(value: Any) -> Optional[str]:
    raw = _compact_spaces(value).translate(ARABIC_DIGITS)
    if not raw:
        return None
    m = re.search(r"(?i)(?:^|[\s<\(\[])([A-Za-z]?\d{3,6})(?:\b|[\s>\)\]])", raw)
    if m:
        normalized = _normalize_reference_code(m.group(1))
        if normalized:
            return normalized
    for token in re.findall(r"[A-Za-z0-9_-]{3,}", raw):
        normalized = _normalize_reference_code(token)
        if normalized:
            return normalized
    return None


def _clean_ocr_label(value: Any) -> str:
    label = _compact_spaces(value)
    if not label:
        return ""
    label = re.sub(r"^[^A-Za-z0-9À-ÿ]+", "", label)
    label = re.sub(r"[^A-Za-z0-9À-ÿ]+$", "", label)
    return _compact_spaces(label)


def _guess_product_number_from_reference(reference: Optional[str]) -> Optional[str]:
    ref = str(reference or "")
    digits = re.sub(r"\D", "", ref)
    if not digits:
        return None
    tail2 = digits[-2:]
    candidate = tail2.lstrip("0")
    if not candidate:
        candidate = digits[-1:]
    if not candidate:
        return None
    return candidate


def _token_is_numeric_tail(token: str) -> bool:
    txt = _compact_spaces(token).translate(ARABIC_DIGITS)
    if not txt:
        return False
    if _is_probable_code(txt):
        return False
    if any(ch.isalpha() for ch in txt):
        return False
    if _to_amount(txt) is not None:
        return True
    if _to_quantity(txt) is not None:
        return True
    return bool(re.fullmatch(r"[+\-]?\d[\d.,/%]*", txt))


def _merge_numeric_tokens(tokens: List[str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(tokens):
        cur = str(tokens[i]).strip()
        if not cur:
            i += 1
            continue
        if i + 1 < len(tokens):
            nxt = str(tokens[i + 1]).strip()
            candidate = f"{cur} {nxt}".strip()
            if candidate and _to_amount(candidate) is not None and (
                _to_amount(cur) is None or len(re.sub(r"\D", "", cur)) <= 2
            ):
                merged.append(candidate)
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged


def _split_dense_numeric_tail(raw: str) -> List[str]:
    tokens = [t for t in str(raw or "").strip().split() if t]
    if len(tokens) < 2:
        return []

    tail_start = len(tokens)
    i = len(tokens) - 1
    while i >= 0 and _token_is_numeric_tail(tokens[i]):
        tail_start = i
        i -= 1

    if tail_start <= 0 or tail_start >= len(tokens):
        return []

    left = " ".join(tokens[:tail_start]).strip()
    if not left:
        return []

    tail_tokens = _merge_numeric_tokens(tokens[tail_start:])
    if not tail_tokens:
        return []
    return [left] + tail_tokens


def _split_dense_header_cells(raw: str) -> List[str]:
    tokens = [t for t in str(raw or "").strip().split() if t]
    if len(tokens) < 3:
        return []
    norm = _norm_text(raw)
    hint_hits = sum(1 for hint in TABLE_HINT_TERMS if hint in norm)
    if hint_hits < 2:
        return []

    synonym_norms = {
        _norm_text(syn)
        for values in FIELD_SYNONYMS.values()
        for syn in values
    }
    out: List[str] = []
    i = 0
    while i < len(tokens):
        cur = tokens[i]
        if i + 1 < len(tokens):
            pair = f"{tokens[i]} {tokens[i + 1]}"
            if _norm_text(pair) in synonym_norms:
                out.append(pair)
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def _is_numeric_like(value: str) -> bool:
    txt = _compact_spaces(value).translate(ARABIC_DIGITS)
    if not txt:
        return False
    if _is_probable_code(txt):
        return False
    if any(ch.isalpha() for ch in txt):
        return False
    txt = txt.replace(",", ".")
    txt = re.sub(r"[^0-9.+\-]", "", txt)
    if not txt:
        return False
    return bool(re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", txt))


def _to_amount(value: str) -> Optional[str]:
    raw = _compact_spaces(value).translate(ARABIC_DIGITS)
    if not raw:
        return None
    if _is_probable_code(raw):
        return None
    norm_raw = _norm_text(raw)
    has_alpha = any(ch.isalpha() for ch in raw)
    currency_hint = bool(re.search(r"(?i)\b(?:dzd|da|eur|usd|tnd|mad|chf|gbp|sar|aed)\b", raw))
    semantic_hint = any(
        k in norm_raw
        for k in (
            "TOTAL",
            "MONTANT",
            "AMOUNT",
            "A PAYER",
            "TVA",
            "VAT",
            "TAX",
            "HT",
            "TTC",
            "PRIX",
            "PRICE",
            "UNIT",
            "TIMBRE",
            "STAMP",
        )
    )
    if has_alpha and not (currency_hint or semantic_hint):
        return None
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", raw) and not semantic_hint:
        return None
    if "/" in raw and not (currency_hint or semantic_hint):
        return None
    if raw.count("-") >= 2 and not (currency_hint or semantic_hint):
        return None
    raw = re.sub(r"(?i)\b(?:dzd|da|eur|usd|tnd|mad|chf|gbp|sar|aed|tva|ht|ttc)\b", " ", raw)
    candidates = re.findall(r"[+\-]?\d[\d\s.,]*", raw)
    if not candidates:
        return None

    def _normalize_number_token(token: str) -> Optional[float]:
        s = str(token or "").strip().replace(" ", "")
        if not s:
            return None
        sign = -1.0 if s.startswith("-") else 1.0
        s = s.lstrip("+-")
        if not s:
            return None
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            if s.count(",") == 1 and len(s.split(",")[-1]) <= 2:
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "." in s and s.count(".") > 1:
            if len(s.split(".")[-1]) <= 2:
                parts = s.split(".")
                s = "".join(parts[:-1]) + "." + parts[-1]
            else:
                s = s.replace(".", "")
        try:
            return sign * float(s)
        except Exception:
            return None

    amount = None
    for candidate in reversed(candidates):
        amount = _normalize_number_token(candidate)
        if amount is not None:
            break
    if amount is None:
        return None
    try:
        return f"{float(amount):.2f}"
    except Exception:
        return None


def _to_quantity(value: str) -> Optional[str]:
    txt = _compact_spaces(value).translate(ARABIC_DIGITS)
    if _is_probable_code(txt):
        return None
    if any(ch.isalpha() for ch in txt):
        return None
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", txt):
        return None
    if "/" in txt:
        return None
    if txt.count("-") >= 2:
        return None
    txt = txt.replace(",", ".")
    txt = re.sub(r"[^0-9.+\-]", "", txt)
    if not re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", txt):
        return None
    try:
        q = float(txt)
        if abs(q - round(q)) < 1e-6:
            return str(int(round(q)))
        return f"{q:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return None


def _header_score(cell_norm: str, field: str, profile: str) -> float:
    best = 0.0
    synonyms = FIELD_SYNONYMS.get(field) or []
    for syn in synonyms:
        syn_norm = _norm_text(syn)
        if not syn_norm:
            continue
        if cell_norm == syn_norm:
            best = max(best, 1.0)
            continue
        if syn_norm in cell_norm:
            best = max(best, 0.93)
            continue
        if cell_norm in syn_norm and len(cell_norm) >= 3:
            best = max(best, 0.85)
            continue
        if profile == "100ml":
            ratio = SequenceMatcher(None, cell_norm, syn_norm).ratio()
            token_overlap = 0.0
            a = set(cell_norm.split())
            b = set(syn_norm.split())
            if a and b:
                token_overlap = len(a.intersection(b)) / float(max(len(a), len(b)))
            best = max(best, 0.75 * ratio + 0.25 * token_overlap)
    return best


def _detect_header_map(cells: List[str], profile: str) -> Tuple[Dict[str, int], float]:
    field_to_idx: Dict[str, int] = {}
    total_score = 0.0
    used_cols = set()
    threshold = 0.82 if profile == "50ml" else 0.70

    for field in ("reference", "product", "quantity", "unit_price", "total", "total_ht", "total_ttc", "tax"):
        best_idx = -1
        best_score = 0.0
        for i, cell in enumerate(cells):
            if i in used_cols:
                continue
            score = _header_score(_norm_text(cell), field, profile)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and best_score >= threshold:
            field_to_idx[field] = best_idx
            used_cols.add(best_idx)
            total_score += best_score

    return field_to_idx, total_score


def _count_norm_hits(norm: str, hints: set[str]) -> int:
    return sum(1 for hint in hints if hint and hint in norm)


def _line_header_hint_score(line: str) -> int:
    norm = _norm_text(line)
    strong = _count_norm_hits(norm, HEADER_STRONG_HINTS)
    weak = _count_norm_hits(norm, HEADER_WEAK_HINTS)
    return (strong * 2) + min(weak, 2)


def _line_footer_hint_score(line: str) -> int:
    norm = _norm_text(line)
    return _count_norm_hits(norm, TOTALS_STOP_HINTS) + _count_norm_hits(norm, FOOTER_ONLY_HINTS)


def _is_totals_or_footer_label(value: Any) -> bool:
    norm = _norm_text(value)
    if not norm:
        return False
    if _count_norm_hits(norm, FOOTER_ONLY_HINTS) >= 1:
        return True
    if _count_norm_hits(norm, TOTALS_STOP_HINTS) >= 1:
        return True
    return False


def _is_footer_like_line(line: str, cells: Optional[List[str]] = None) -> bool:
    norm = _norm_text(line)
    footer_hits = _line_footer_hint_score(line)
    header_strong_hits = _count_norm_hits(norm, HEADER_STRONG_HINTS)
    amountish_hits = sum(1 for hint in ("TOTAL", "MONTANT", "AMOUNT", "TAX", "TVA", "VAT", "TIMBRE", "STAMP", "HT", "TTC") if hint in norm)
    cell_list = cells if isinstance(cells, list) else []

    if footer_hits >= 2:
        return True
    if footer_hits >= 1 and header_strong_hits == 0:
        return True
    if amountish_hits >= 2 and header_strong_hits == 0 and len(cell_list) <= 2:
        return True
    if any(phrase in norm for phrase in ("NON ASSUJETTI", "MONTANT A PAYER", "TOTAL DUE", "AMOUNT DUE", "TERMS CONDITIONS", "DATA INFO", "FACTURE EN LETTRE", "MODE DE PAIEMENT", "SIGNATURE", "CACHET")):
        return True
    return False


def _is_header_like_line(line: str, cells: List[str]) -> bool:
    score = _line_header_hint_score(line)
    strong_hits = _count_norm_hits(_norm_text(line), HEADER_STRONG_HINTS)
    if _is_footer_like_line(line, cells):
        return False
    if strong_hits >= 2:
        return True
    if score >= 4:
        return True
    return score >= 3 and len(cells) >= 3


def _line_looks_tabular(cells: List[str], line: str) -> bool:
    if not cells:
        return False
    if _is_footer_like_line(line, cells) and not _is_header_like_line(line, cells):
        return False
    if len(cells) == 1 and _is_probable_code(str(cells[0])):
        return True
    if len(cells) >= 4:
        return True

    numeric = sum(1 for c in cells if _to_amount(c) is not None or _to_quantity(c) is not None)
    texty = sum(1 for c in cells if not _is_numeric_like(c))
    if numeric >= 2 and not _is_footer_like_line(line, cells):
        return True
    if _is_header_like_line(line, cells):
        return True
    if len(cells) == 3 and numeric >= 1 and texty >= 1 and not _is_footer_like_line(line, cells):
        return True
    if len(cells) == 2 and numeric >= 1:
        left, right = cells[0], cells[1]
        if _is_probable_code(left) and _to_amount(right) is not None:
            return True
        right_amount = _to_amount(right) is not None
        left_norm = _norm_text(left)
        left_forbidden = any(hint in left_norm for hint in NON_TABLE_LEFT_HINTS) or _is_totals_or_footer_label(left)
        if right_amount and texty >= 1 and ":" not in left and not left_forbidden:
            return True
        return False
    n = _norm_text(line)
    return (not _is_footer_like_line(line, cells)) and any(k in n for k in ("QTE", "QTY", "QNTY", "PRIX", "PRICE", "PRODUIT", "PRODUCT", "ARTICLE", "ITEM", "ITEMS"))


def _is_strong_single_row(row: Dict[str, Any]) -> bool:
    cells = row.get("cells") or []
    if not isinstance(cells, list) or not cells:
        return False
    line = str(row.get("line") or "")
    numeric = sum(1 for c in cells if _to_amount(c) is not None or _to_quantity(c) is not None)
    if numeric >= 2:
        return True
    if len(cells) >= 2 and numeric >= 1 and _is_probable_code(str(cells[0])):
        return True
    return _is_header_like_line(line, [str(c) for c in cells])


def _collect_blocks(lines: List[str]) -> List[List[Dict[str, Any]]]:
    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    pending_header: Optional[Dict[str, Any]] = None
    idle = 0
    for line in lines:
        cells = _split_line_cells(line)
        header_like = _is_header_like_line(line, cells)
        footer_like = _is_footer_like_line(line, cells)
        is_tab = _line_looks_tabular(cells, line)
        if footer_like and current:
            if len(current) >= 2 or (len(current) == 1 and _is_strong_single_row(current[0])):
                blocks.append(current)
            current = []
            pending_header = None
            idle = 0
            continue
        if is_tab:
            if not current and pending_header and pending_header.get("line") != line:
                current.append(pending_header)
            current.append({"line": line, "cells": cells})
            pending_header = {"line": line, "cells": cells} if header_like else None
            idle = 0
            continue
        if header_like and len(cells) >= 2:
            pending_header = {"line": line, "cells": cells}
        if current:
            idle += 1
            if idle <= 1:
                continue
            if len(current) >= 2 or (len(current) == 1 and _is_strong_single_row(current[0])):
                blocks.append(current)
            current = []
            idle = 0
    if len(current) >= 2 or (len(current) == 1 and _is_strong_single_row(current[0])):
        blocks.append(current)
    return blocks


def _infer_map_from_rows(rows: List[List[str]]) -> Dict[str, int]:
    if not rows:
        return {}
    max_cols = max(len(r) for r in rows)
    if max_cols < 2:
        return {}

    if max_cols >= 5:
        pattern_total = 0
        pattern_hits = 0
        for row in rows:
            if len(row) < 5:
                continue
            pattern_total += 1
            ref_ok = _normalize_reference_code(row[0]) is not None
            prod_ok = not _is_numeric_like(row[1]) and len(_compact_spaces(row[1])) >= 2
            qty_ok = _to_quantity(row[2]) is not None
            unit_ok = _to_amount(row[3]) is not None
            total_ok = _to_amount(row[4]) is not None
            if ref_ok and prod_ok and qty_ok and unit_ok and total_ok:
                pattern_hits += 1
        if pattern_hits >= 2 and pattern_hits >= int(max(1, pattern_total * 0.4)):
            return {"reference": 0, "product": 1, "quantity": 2, "unit_price": 3, "total": 4}

    if max_cols >= 4:
        pattern_total = 0
        pattern_hits = 0
        for row in rows:
            if len(row) < 4:
                continue
            pattern_total += 1
            qty_ok = _to_quantity(row[0]) is not None
            prod_ok = not _is_numeric_like(row[1]) and len(_compact_spaces(row[1])) >= 3
            unit_ok = _to_amount(row[2]) is not None
            total_ok = _to_amount(row[3]) is not None
            if qty_ok and prod_ok and unit_ok and total_ok:
                pattern_hits += 1
        if pattern_hits >= 2 and pattern_hits >= int(max(1, pattern_total * 0.4)):
            return {"quantity": 0, "product": 1, "unit_price": 2, "total": 3}

    numeric_ratio: List[float] = []
    code_ratio: List[float] = []
    for i in range(max_cols):
        total = 0
        numeric = 0
        code_like = 0
        for row in rows:
            if i >= len(row):
                continue
            total += 1
            cell = row[i]
            if _to_amount(cell) is not None or _to_quantity(cell) is not None:
                numeric += 1
            if _normalize_reference_code(cell):
                code_like += 1
        numeric_ratio.append(float(numeric) / float(max(1, total)))
        code_ratio.append(float(code_like) / float(max(1, total)))

    mapping: Dict[str, int] = {}
    ref_candidates = [i for i, r in enumerate(code_ratio) if r >= 0.6]
    if ref_candidates:
        mapping["reference"] = max(ref_candidates, key=lambda idx: code_ratio[idx])
    product_candidates = [i for i, r in enumerate(numeric_ratio) if r < 0.45]
    if "reference" in mapping:
        product_candidates = [i for i in product_candidates if i != mapping["reference"]]
    if product_candidates:
        best_idx = product_candidates[0]
        best_alpha = -1.0
        for idx in product_candidates:
            alpha_chars = 0
            token_rows = 0
            for row in rows:
                if idx >= len(row):
                    continue
                cell = _compact_spaces(row[idx])
                if not cell:
                    continue
                token_rows += 1
                alpha_chars += sum(1 for ch in cell if ch.isalpha())
            alpha_ratio = float(alpha_chars) / float(max(1, token_rows))
            if alpha_ratio > best_alpha:
                best_alpha = alpha_ratio
                best_idx = idx
        mapping["product"] = best_idx
    else:
        mapping["product"] = 0

    numeric_cols = [i for i, r in enumerate(numeric_ratio) if r >= 0.55]
    if numeric_cols:
        mapping["total"] = numeric_cols[-1]
        if len(numeric_cols) >= 2:
            mapping["unit_price"] = numeric_cols[-2]
        if len(numeric_cols) >= 3:
            mapping["quantity"] = numeric_cols[0]
    return mapping


def _extract_totals_from_line(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    norm = _norm_text(line)
    amount = _to_amount(line)
    if not amount:
        return out

    if "TOTAL TTC" in norm or "MONTANT A PAYER TTC" in norm or ("TTC" in norm and "TOTAL" in norm):
        out["total_ttc"] = amount
        return out

    if "TOTAL HT" in norm or "MONTANT HT" in norm or "SUBTOTAL" in norm or "SOUS TOTAL" in norm or ("HT" in norm and "TOTAL" in norm):
        out["total_ht"] = amount
        return out

    if "TIMBRE" in norm or "STAMP" in norm:
        out["stamp"] = amount
        return out

    if "MONTANT A PAYER" in norm or "TOTAL A PAYER" in norm or "AMOUNT DUE" in norm or "TOTAL DUE" in norm:
        out["amount_due"] = amount
        return out

    if "TVA" in norm or "VAT" in norm or "TAX" in norm or "TAXES" in norm:
        out["tax"] = amount
        return out

    if "TOTAL" in norm or "MONTANT" in norm or "AMOUNT" in norm:
        out["total"] = amount

    return out


def _row_to_item(
    cells: List[str],
    col_map: Dict[str, int],
    table_index: int,
    page_index: int,
    row_index: int,
    profile: str,
) -> Optional[Dict[str, Any]]:
    def get(field: str) -> str:
        idx = col_map.get(field, -1)
        if idx < 0 or idx >= len(cells):
            return ""
        return _compact_spaces(cells[idx])

    reference = _normalize_reference_code(get("reference"))
    product = get("product")
    quantity = _to_quantity(get("quantity"))
    unit_price = _to_amount(get("unit_price"))
    total_ht = _to_amount(get("total_ht"))
    total_ttc = _to_amount(get("total_ttc"))
    total = _to_amount(get("total"))

    if len(cells) == 2 and _is_probable_code(str(cells[0])):
        amount_tail = _to_amount(str(cells[1]))
        if amount_tail:
            if not unit_price:
                unit_price = amount_tail
            if not total:
                total = amount_tail
        reference = reference or _normalize_reference_code(cells[0])

    if not product:
        non_numeric = [c for c in cells if _to_amount(c) is None and _to_quantity(c) is None]
        if non_numeric:
            product = max(non_numeric, key=lambda x: len(x))
    if product and _is_numeric_like(product):
        non_numeric = [c for c in cells if not _is_numeric_like(c)]
        if non_numeric:
            product = max(non_numeric, key=lambda x: len(x))
    if not reference:
        for candidate in cells:
            reference = _extract_reference_code(candidate)
            if reference:
                break
    if reference and product:
        if _normalize_reference_code(product) == reference:
            product = ""

    joined_line = " | ".join(str(c) for c in cells if _compact_spaces(c))

    if _is_totals_or_footer_label(joined_line):
        return None
    if _is_header_like_line(joined_line, cells):
        return None
    if _is_footer_like_line(joined_line, cells):
        return None
    if any(_is_totals_or_footer_label(c) for c in cells):
        if len(cells) <= 3:
            return None
    if product:
        product_norm = _norm_text(product)
        if any(hint in product_norm for hint in NON_TABLE_LEFT_HINTS) or _is_totals_or_footer_label(product):
            return None

    if profile == "100ml" and not unit_price and not total and len(cells) >= 3:
        numeric_cells = [(i, _to_amount(c)) for i, c in enumerate(cells)]
        numeric_cells = [(i, v) for i, v in numeric_cells if v is not None]
        if len(numeric_cells) >= 2:
            unit_price = unit_price or numeric_cells[-2][1]
            total = total or numeric_cells[-1][1]
        if not quantity:
            q_candidates = [c for c in cells if _to_quantity(c) is not None]
            if q_candidates:
                quantity = _to_quantity(q_candidates[0])

    has_any = bool(reference or product or quantity or unit_price or total_ht or total_ttc or total)
    if not has_any:
        return None
    has_product_identity = bool(reference or product)
    has_complete_pricing = quantity is not None and unit_price is not None
    if not has_product_identity:
        return None
    if not has_complete_pricing:
        return None
    has_numeric_value = bool(quantity or unit_price or total_ht or total_ttc or total)
    if not has_numeric_value:
        first_cell = str(cells[0]) if cells else ""
        if not _is_probable_code(first_cell):
            return None
    if not reference and len(cells) <= 2 and any(_to_amount(c) is not None for c in cells) and any(_is_totals_or_footer_label(c) for c in cells):
        return None

    score = 0.0
    if reference:
        score += 0.12
    if product:
        score += 0.35
    if quantity:
        score += 0.2
    if unit_price:
        score += 0.2
    if total or total_ht or total_ttc:
        score += 0.25
    if profile == "100ml":
        score = min(1.0, score + 0.05)

    return {
        "table_index": table_index,
        "page_index": page_index,
        "row_index": row_index,
        "reference": reference,
        "product": product or None,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_ht": total_ht,
        "total_ttc": total_ttc,
        "total": total,
        "raw_cells": cells,
        "confidence": round(score, 4),
    }


def _is_complete_line_item(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    reference = _normalize_reference_code(row.get("reference"))
    product = _compact_spaces(row.get("product"))
    label = product or reference or ""
    if not label:
        return False
    if _is_totals_or_footer_label(label):
        return False
    quantity = _to_quantity(row.get("quantity"))
    unit_price = _to_amount(row.get("unit_price"))
    return quantity is not None and unit_price is not None


def _filter_complete_line_items(extracted: Dict[str, Any]) -> Dict[str, Any]:
    tables = [t for t in _safe_list(extracted.get("tables")) if isinstance(t, dict)]
    totals = extracted.get("totals") if isinstance(extracted.get("totals"), dict) else {}
    kept_tables: List[Dict[str, Any]] = []
    rebuilt_rows: List[Dict[str, Any]] = []

    for table in tables:
        table_type = str(table.get("table_type") or "")
        if table_type != "line_items":
            continue
        rows = [r for r in _safe_list(table.get("rows")) if isinstance(r, dict)]
        rows = [r for r in rows if _is_complete_line_item(r)]
        if not rows:
            continue
        new_table = dict(table)
        new_table["rows"] = rows
        new_table["rows_count"] = len(rows)
        kept_tables.append(new_table)

    for new_idx, table in enumerate(kept_tables, start=1):
        table["table_index"] = new_idx
        rows = [r for r in _safe_list(table.get("rows")) if isinstance(r, dict)]
        for row_idx, row in enumerate(rows, start=1):
            row["table_index"] = new_idx
            row["row_index"] = row_idx
        table["rows"] = rows
        rebuilt_rows.extend(rows)

    dedup: List[Dict[str, Any]] = []
    seen = set()
    for row in rebuilt_rows:
        sig = (
            str(row.get("page_index") or ""),
            str(row.get("reference") or ""),
            _compact_spaces(row.get("product")).upper(),
            str(row.get("quantity") or ""),
            str(row.get("unit_price") or ""),
            str(row.get("total_ht") or ""),
            str(row.get("total_ttc") or ""),
            str(row.get("total") or ""),
        )
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append(row)

    return {
        "tables_count": len(kept_tables),
        "rows_total": len(dedup),
        "detected_columns": _infer_detected_columns_from_rows(dedup),
        "totals": totals,
        "tables": kept_tables,
        "line_items": dedup,
    }


def _extract_doc_tables(doc: Dict[str, Any], profile: str) -> Dict[str, Any]:
    tables: List[Dict[str, Any]] = []
    line_items: List[Dict[str, Any]] = []
    detected_columns = set()
    totals: Dict[str, List[str]] = {
    "total_ht": [],
    "total_ttc": [],
    "tax": [],
    "total": [],
    "amount_due": [],
    "stamp": [],
}

    pages = _safe_list(doc.get("pages"))
    table_index = 0

    for page in pages:
        if not isinstance(page, dict):
            continue
        page_index = int(page.get("page_index") or page.get("page") or 1)
        lines: List[str] = []

        page_text = str(page.get("page_text") or page.get("ocr_text") or page.get("text") or "")
        if page_text:
            lines.extend([str(ln).replace("\r", "") for ln in page_text.splitlines()])

        for sent in _safe_list(page.get("sentences_layout")):
            if not isinstance(sent, dict):
                continue
            kind = str(sent.get("layout_kind") or "")
            if kind not in {"table", "multicol_grid", "header"}:
                continue
            txt = _compact_spaces(sent.get("text") or "")
            if txt and (kind in {"table", "multicol_grid"} or _line_header_hint_score(txt) >= 2):
                lines.append(txt)
            if kind in {"table", "multicol_grid"}:
                for hdr_row in _safe_list(sent.get("header_rows")):
                    if not isinstance(hdr_row, list):
                        continue
                    hdr_cells: List[str] = []
                    for cell in hdr_row:
                        if isinstance(cell, dict):
                            val = _compact_spaces(cell.get("text") or "")
                        else:
                            val = _compact_spaces(cell)
                        if val:
                            hdr_cells.append(val)
                    if len(hdr_cells) >= 2:
                        lines.append(" | ".join(hdr_cells))
                for row in _safe_list(sent.get("table_rows")):
                    if isinstance(row, dict):
                        row_txt = _compact_spaces(row.get("text") or "")
                        if row_txt:
                            lines.append(row_txt)
                        row_cells = _safe_list(row.get("cells"))
                        if row_cells:
                            cell_values = [_compact_spaces(c) for c in row_cells if _compact_spaces(c)]
                            if len(cell_values) >= 2:
                                lines.append(" | ".join(cell_values))

        if not lines:
            continue

        seen = set()
        unique_lines: List[str] = []
        for line in lines:
            k = _compact_spaces(line)
            if not k or k in seen:
                continue
            seen.add(k)
            unique_lines.append(str(line).rstrip("\r\n"))

        anchor_blocks = _collect_blocks_anchor(lines)
        classic_blocks = _collect_blocks(unique_lines)
        blocks = _merge_blocks(anchor_blocks, classic_blocks)
        for block in blocks:
            if not block:
                continue
            table_index += 1
            block_cells = [b.get("cells") or [] for b in block if isinstance(b, dict)]
            columns_estimated = max((len(cells) for cells in block_cells if isinstance(cells, list)), default=0)
            header_idx = 0
            header_map, header_score = _detect_header_map(block_cells[0] if block_cells else [], profile)
            if len(header_map) < 2 and len(block_cells) >= 2:
                hm2, hs2 = _detect_header_map(block_cells[1], profile)
                if len(hm2) > len(header_map) or hs2 > header_score:
                    header_map, header_score = hm2, hs2
                    header_idx = 1

            data_rows = block_cells[header_idx + 1 :] if len(header_map) >= 2 else block_cells
            if len(header_map) < 2:
                header_map = _infer_map_from_rows(data_rows[:10])

            for field in header_map.keys():
                detected_columns.add(field)

            parsed_rows: List[Dict[str, Any]] = []
            row_idx = 0
            for cells in data_rows:
                row_idx += 1
                item = _row_to_item(cells, header_map, table_index, page_index, row_idx, profile)
                if item:
                    parsed_rows.append(item)
                    line_items.append(item)
                    if item.get("reference"):
                        detected_columns.add("reference")
                    if item.get("product"):
                        detected_columns.add("product")
                    if item.get("quantity"):
                        detected_columns.add("quantity")
                    if item.get("unit_price"):
                        detected_columns.add("unit_price")
                    if item.get("total") or item.get("total_ht") or item.get("total_ttc"):
                        detected_columns.add("total")

            if not parsed_rows:
                continue

            tables.append(
                {
                    "table_index": table_index,
                    "page_index": page_index,
                    "table_type": "line_items",
                    "header_map": header_map,
                    "header_score": round(float(header_score), 4),
                    "rows_count": len(parsed_rows),
                    "shape": {
                        "source": "layout-lines",
                        "columns_estimated": int(columns_estimated),
                        "rows_estimated": int(len(parsed_rows)),
                    },
                    "rows": parsed_rows,
                }
            )

        for line in unique_lines:
            t = _extract_totals_from_line(line)
            for k, v in t.items():
                if k not in totals or not isinstance(totals.get(k), list):
                    totals[k] = []
                if v and v not in totals[k]:
                    totals[k].append(v)

    table_index = _augment_code_only_rows_from_header(doc, tables, line_items, table_index)
    table_index = _augment_text_line_items_from_header(
        doc,
        tables,
        line_items,
        table_index,
        detected_columns,
        profile,
    )

    dedup: List[Dict[str, Any]] = []
    seen_items = set()
    for item in line_items:
        sig = (
            str(item.get("page_index")),
            str(item.get("reference")),
            str(item.get("product")),
            str(item.get("quantity")),
            str(item.get("unit_price")),
            str(item.get("total_ht")),
            str(item.get("total_ttc")),
            str(item.get("total")),
        )
        if sig in seen_items:
            continue
        seen_items.add(sig)
        dedup.append(item)

    return {
        "tables_count": len(tables),
        "rows_total": len(dedup),
        "detected_columns": sorted(detected_columns),
        "totals": totals,
        "tables": tables,
        "line_items": dedup,
    }


def _augment_code_only_rows_from_header(
    doc: Dict[str, Any],
    tables: List[Dict[str, Any]],
    line_items: List[Dict[str, Any]],
    table_index: int,
) -> int:
    seen_products = {
        _compact_spaces(item.get("product")).upper()
        for item in line_items
        if isinstance(item, dict) and _compact_spaces(item.get("product"))
    }
    pages = _safe_list(doc.get("pages"))
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_index = int(page.get("page_index") or page.get("page") or 1)
        text = str(page.get("page_text") or page.get("ocr_text") or page.get("text") or "")
        if not text:
            continue
        lines = [_compact_spaces(ln) for ln in text.splitlines() if _compact_spaces(ln)]
        if not lines:
            continue

        header_pos = -1
        for i, line in enumerate(lines):
            norm = _norm_text(line)
            if _line_header_hint_score(line) >= 2 and (
                "REFERENCE" in norm or "PRODUIT" in norm or "ARTICLE" in norm or "DESIGNATION" in norm
            ):
                header_pos = i
                break
        if header_pos < 0:
            continue

        page_new_rows: List[Dict[str, Any]] = []
        row_idx = 0
        for line in lines[header_pos + 1 :]:
            norm = _norm_text(line)
            if any(stop in norm for stop in ("TOTAL", "TVA", "TTC", "HT", "CACHET", "SIGNATURE", "SUBTOTAL", "TAX", "TAXES", "MONTANT A PAYER", "AMOUNT DUE", "TIMBRE")):
                break
            codes = re.findall(r"\b[A-Za-z]{0,4}\d{3,}\b", line)
            if not codes:
                continue
            for code in codes:
                code_norm = _compact_spaces(code).upper()
                if not code_norm or not _is_probable_code(code_norm):
                    continue
                if code_norm in seen_products:
                    continue
                seen_products.add(code_norm)
                row_idx += 1
                page_new_rows.append(
                    {
                        "table_index": table_index + 1,
                        "page_index": page_index,
                        "row_index": row_idx,
                        "reference": code_norm,
                        "product": code_norm,
                        "quantity": None,
                        "unit_price": None,
                        "total_ht": None,
                        "total_ttc": None,
                        "total": None,
                        "raw_cells": [code_norm],
                        "confidence": 0.45,
                    }
                )

        if not page_new_rows:
            continue
        table_index += 1
        for r_idx, row in enumerate(page_new_rows, start=1):
            row["table_index"] = table_index
            row["row_index"] = r_idx
        line_items.extend(page_new_rows)
        tables.append(
            {
                "table_index": table_index,
                "page_index": page_index,
                "table_type": "codes_only",
                "header_map": {"reference": 0, "product": 0},
                "header_score": 0.0,
                "rows_count": len(page_new_rows),
                "shape": {
                    "source": "layout-header-codes",
                    "columns_estimated": 2,
                    "rows_estimated": int(len(page_new_rows)),
                },
                "rows": page_new_rows,
            }
        )
    return table_index


def _augment_text_line_items_from_header(
    doc: Dict[str, Any],
    tables: List[Dict[str, Any]],
    line_items: List[Dict[str, Any]],
    table_index: int,
    detected_columns: Any,
    profile: str,
) -> int:
    existing_signatures = {
        (
            str(item.get("page_index")),
            str(item.get("product")),
            str(item.get("quantity")),
            str(item.get("unit_price")),
            str(item.get("total")),
        )
        for item in line_items
        if isinstance(item, dict)
    }

    pages = _safe_list(doc.get("pages"))
    for page in pages:
        if not isinstance(page, dict):
            continue

        page_index = int(page.get("page_index") or page.get("page") or 1)

        already_has_line_items = any(
            str(t.get("table_type") or "") == "line_items"
            and int(t.get("page_index") or 0) == page_index
            and int(t.get("rows_count") or 0) >= 3
            for t in tables
            if isinstance(t, dict)
        )
        if already_has_line_items:
            continue

        text = str(page.get("page_text") or page.get("ocr_text") or page.get("text") or "")
        if not text:
            continue

        lines = [_compact_spaces(ln) for ln in text.splitlines() if _compact_spaces(ln)]
        if not lines:
            continue

        header_pos = -1
        header_map: Dict[str, int] = {}
        header_score = 0.0
        header_cells: List[str] = []

        for i, line in enumerate(lines):
            cells = _split_line_cells(line)
            if len(cells) < 3:
                continue

            hm, hs = _detect_header_map(cells, profile)
            norm = _norm_text(line)

            if len(hm) >= 3 and any(
                hint in norm
                for hint in (
                    "ITEM",
                    "ITEMS",
                    "SERVICE",
                    "SERVICES",
                    "DESCRIPTION",
                    "PRODUCT",
                    "PRODUIT",
                    "ARTICLE",
                    "REFERENCE",
                    "QTY",
                    "QNTY",
                    "QUANTITY",
                    "QUANTITE",
                    "PRICE",
                    "PRIX",
                )
            ):
                header_pos = i
                header_map = hm
                header_score = hs
                header_cells = cells
                break

        if header_pos < 0:
            continue

        if len(header_map) < 2:
            header_map = _infer_map_from_rows([_split_line_cells(ln) for ln in lines[header_pos + 1 : header_pos + 8]])

        if "product" not in header_map or "quantity" not in header_map or "unit_price" not in header_map:
            continue

        page_new_rows: List[Dict[str, Any]] = []

        for line in lines[header_pos + 1 :]:
            raw = _compact_spaces(line)
            if not raw:
                continue

            cells = _split_line_cells(raw)
            if not cells:
                continue

            if _extract_totals_from_line(raw):
                break
            if _is_footer_like_line(raw, cells):
                break

            item = _row_to_item(
                cells,
                header_map,
                table_index + 1,
                page_index,
                len(page_new_rows) + 1,
                profile,
            )
            if not item:
                continue

            sig = (
                str(item.get("page_index")),
                str(item.get("product")),
                str(item.get("quantity")),
                str(item.get("unit_price")),
                str(item.get("total")),
            )
            if sig in existing_signatures:
                continue

            existing_signatures.add(sig)
            page_new_rows.append(item)

        if not page_new_rows:
            continue

        table_index += 1

        for idx, row in enumerate(page_new_rows, start=1):
            row["table_index"] = table_index
            row["row_index"] = idx

        tables.append(
            {
                "table_index": table_index,
                "page_index": page_index,
                "table_type": "line_items",
                "header_map": header_map,
                "header_score": round(float(header_score), 4),
                "rows_count": len(page_new_rows),
                "shape": {
                    "source": "text-header-fallback",
                    "columns_estimated": max(len(header_cells), 4),
                    "rows_estimated": len(page_new_rows),
                },
                "rows": page_new_rows,
            }
        )

        line_items.extend(page_new_rows)

        for field in ("product", "quantity", "unit_price", "total", "reference"):
            if field in header_map:
                detected_columns.add(field)

    return table_index


def _build_source_path_map(ctx: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    repo_root = Path(__file__).resolve().parents[2]

    def _add(path_like: Any) -> None:
        path = str(path_like or "").strip()
        if not path:
            return
        p = Path(path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        p = resolve_runtime_input_path(p, repo_root)
        name = p.name.lower()
        if name and name not in out:
            out[name] = str(p)

    for row in _safe_list(ctx.get("PRETRAITEMENT_RESULT")):
        if isinstance(row, dict):
            _add(row.get("path"))
    for path in _safe_list(ctx.get("INPUT_FILE")):
        _add(path)
    for path in _safe_list(ctx.get("IMAGE_ONLY_FILES")):
        _add(path)
    return out


def _should_run_ocr_fallback(extracted: Dict[str, Any], source_path: Optional[str]) -> bool:
    if not source_path:
        return False
    suffix = Path(source_path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return False

    rows = _safe_list(extracted.get("line_items"))
    if not rows:
        return True

    numeric_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("unit_price") or row.get("total") or row.get("total_ht") or row.get("total_ttc"):
            numeric_rows += 1

    if numeric_rows >= 4:
        return False

    tables = [t for t in _safe_list(extracted.get("tables")) if isinstance(t, dict)]
    has_rich_line_items = any(
        str(t.get("table_type") or "") == "line_items" and int(t.get("rows_count") or 0) >= 5
        for t in tables
    )
    if has_rich_line_items:
        return False

    return numeric_rows <= 2


def _parse_ocr_row_line(line: str) -> Optional[Dict[str, Any]]:
    raw = _compact_spaces(line)
    if not raw:
        return None
    if _is_footer_like_line(raw, _split_line_cells(raw)):
        return None

    reference = _extract_reference_code(raw)
    if not reference:
        return None

    product = str(reference)
    prod_match = re.search(
        r"(?i)\b(?:PRODUIT|PRODUCT|ARTICLE|DESIGNATION)\b\s*[:\-]?\s*([A-Za-z0-9#]+)",
        raw,
    )
    if prod_match:
        suffix = _clean_ocr_label(prod_match.group(1))
        if suffix:
            product = f"PRODUIT {suffix}"

    amount_like_tokens = re.findall(r"[+\-]?\d[\d.,]*", raw)
    num_tokens = amount_like_tokens[:]
    if len(num_tokens) < 2:
        return {
            "reference": reference,
            "product": product,
            "quantity": None,
            "unit_price": None,
            "total": None,
            "raw_cells": [raw],
            "confidence": 0.45,
        }

    amount_tokens = [tok for tok in num_tokens if "." in tok or "," in tok]
    if len(amount_tokens) >= 2:
        unit_raw, total_raw = amount_tokens[-2], amount_tokens[-1]
    else:
        unit_raw, total_raw = num_tokens[-2], num_tokens[-1]

    unit_price = _to_amount(unit_raw)
    total = _to_amount(total_raw)
    quantity = None
    if len(num_tokens) >= 3:
        qty_candidate = num_tokens[-3]
        if qty_candidate != re.sub(r"\D", "", str(reference)):
            quantity = _to_quantity(qty_candidate)

    if not prod_match:
        work = raw
        work = re.sub(rf"(?i)\b{re.escape(reference)}\b", " ", work, count=1)
        work = re.sub(r"(?:[+\-]?\d[\d.,]*\s*){2,}$", " ", work)
        cleaned = _clean_ocr_label(work)
        if cleaned and len(cleaned) >= 3 and not _is_probable_code(cleaned):
            product = cleaned

    product = _clean_ocr_label(product) or str(reference)
    product_norm = _norm_text(product)
    if "PRODUIT" in product_norm:
        suffix_num = _guess_product_number_from_reference(reference)
        has_digit = bool(re.search(r"\d", product))
        if suffix_num and not has_digit:
            product = f"PRODUIT {suffix_num}"

    if not (unit_price or total):
        return {
            "reference": reference,
            "product": product,
            "quantity": quantity,
            "unit_price": None,
            "total": None,
            "raw_cells": [raw],
            "confidence": 0.45,
        }
    return {
        "reference": reference,
        "product": product,
        "quantity": quantity,
        "unit_price": unit_price,
        "total": total,
        "raw_cells": [raw],
        "confidence": 0.62,
    }


def _parse_ocr_table_rows(text: str) -> List[Dict[str, Any]]:
    lines = [_compact_spaces(ln) for ln in str(text or "").splitlines() if _compact_spaces(ln)]
    if not lines:
        return []

    out: List[Dict[str, Any]] = []
    table_group = 0
    in_table = False
    for line in lines:
        norm = _norm_text(line)
        if (
            ("REFERENCE" in norm or "REF" in norm)
            and ("PRODUIT" in norm or "PRODUCT" in norm or "DESCRIPTION" in norm or "ARTICLE" in norm)
        ) or (
            ("QUANTITE" in norm or "QTE" in norm or "QTY" in norm)
            and (
                "P UNITAIRE" in norm
                or "P.UNITAIRE" in norm
                or "PRIX" in norm
                or "VALEUR" in norm
                or "TOTAL" in norm
            )
        ):
            table_group += 1
            in_table = True
            continue
        if not in_table:
            continue
        if any(
            stop in norm
            for stop in (
                "NON ASSUJETTI",
                "MONTANT A PAYER",
                "MONTANT FACTURE",
                "MODE DE PAIEMENT",
                "CACHET",
                "SIGNATURE",
                "FACTURE EN LETTRE",
                "SUBTOTAL",
                "SOUS TOTAL",
                "TAX",
                "TAXES",
                "AMOUNT DUE",
                "TOTAL DUE",
                "TIMBRE",
            )
        ) or _is_footer_like_line(line, _split_line_cells(line)):
            in_table = False
            continue
        row = _parse_ocr_row_line(line)
        if not row:
            continue
        row["_group"] = max(1, table_group)
        out.append(row)
    return out


def _guess_total_field_key(label: str) -> str:
    norm = _norm_text(label)
    for hints, field_key in TOTAL_FIELD_HINTS:
        if any(h in norm for h in hints):
            return field_key
    return "total"


def _parse_ocr_totals_rows(text: str) -> List[Dict[str, Any]]:
    lines = [_compact_spaces(ln) for ln in str(text or "").splitlines() if _compact_spaces(ln)]
    out: List[Dict[str, Any]] = []
    if not lines:
        return out

    for line in lines:
        amount = _to_amount(line)
        if not amount:
            continue
        norm = _norm_text(line)
        if not any(
            hint in norm
            for hint in (
                "TOTAL",
                "SUBTOTAL",
                "SOUS TOTAL",
                "MONTANT",
                "TVA",
                "VAT",
                "TAX",
                "TAXES",
                "TIMBRE",
                "STAMP",
                "A PAYER",
                "AMOUNT DUE",
                "TOTAL DUE",
            )
        ):
            continue
        label = re.sub(r"[+\-]?\d[\d\s.,]*", " ", line)
        label = _clean_ocr_label(label.strip(":- "))
        if not label:
            continue
        key = _guess_total_field_key(label)
        out.append(
            {
                "label": label,
                "field_key": key,
                "value": amount,
                "raw_cells": [label, amount],
                "confidence": 0.7,
            }
        )

    dedup: List[Dict[str, Any]] = []
    seen = set()
    for row in out:
        sig = (_norm_text(row.get("label")), str(row.get("value")))
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append(row)
    return dedup


def _ocr_fallback_tables_from_image(source_path: str) -> Dict[str, Any]:
    try:
        from PIL import Image, ImageFilter, ImageOps
        import pytesseract
    except Exception:
        return {"line_rows": [], "totals_rows": []}

    repo_root = Path(__file__).resolve().parents[2]
    path = resolve_runtime_input_path(Path(source_path), repo_root)
    if not path.exists():
        return {"line_rows": [], "totals_rows": []}

    try:
        image = Image.open(path)
    except Exception:
        return {"line_rows": [], "totals_rows": []}

    def _build_variants(img: Any) -> List[Any]:
        variants = [img]
        try:
            gray = ImageOps.grayscale(img)
            enhanced = ImageOps.autocontrast(gray)
            if min(enhanced.size) < 1200:
                enhanced = enhanced.resize((enhanced.size[0] * 2, enhanced.size[1] * 2))
            bw = enhanced.point(lambda p: 255 if p > 168 else 0)
            sharp = enhanced.filter(ImageFilter.SHARPEN)
            variants.extend([enhanced, bw, sharp])
        except Exception:
            pass
        out: List[Any] = []
        for v in variants:
            if v is None:
                continue
            out.append(v)
        return out

    variants = _build_variants(image)
    best_line_rows: List[Dict[str, Any]] = []
    best_totals_rows: List[Dict[str, Any]] = []
    best_score = -1
    for variant in variants:
        for config in ("--psm 3", "--psm 4", "--psm 6 -c preserve_interword_spaces=1"):
            try:
                text = pytesseract.image_to_string(variant, lang="fra+eng", config=config)
            except Exception:
                continue
            rows = _parse_ocr_table_rows(text)
            totals_rows = _parse_ocr_totals_rows(text)
            numeric = sum(1 for r in rows if r.get("unit_price") or r.get("total"))
            qty = sum(1 for r in rows if r.get("quantity"))
            refs = {str(r.get("reference") or "") for r in rows if str(r.get("reference") or "")}
            ref_quality = sum(1 for r in rows if re.fullmatch(r"[A-Z]\d{3,6}", str(r.get("reference") or "")))
            product_quality = sum(
                1
                for r in rows
                if re.search(r"(?i)\bPRODUIT\s+\d+\b", str(r.get("product") or ""))
            )
            amount_quality = sum(
                1
                for r in rows
                if (r.get("unit_price") and r.get("total"))
                and float(str(r.get("total")).replace(",", ".")) >= float(str(r.get("unit_price")).replace(",", "."))
            )
            noise = sum(
                1
                for r in rows
                if any(ch in str(r.get("product") or "") for ch in ("|", "[", "]", "?", "_"))
            )
            score = (
                numeric * 4
                + len(rows) * 2
                + len(refs) * 2
                + len(totals_rows)
                + ref_quality * 4
                + product_quality * 3
                + qty
                + amount_quality
                - noise * 2
            )
            if score > best_score:
                best_score = score
                best_line_rows = rows
                best_totals_rows = totals_rows
    return {"line_rows": best_line_rows, "totals_rows": best_totals_rows}


def _merge_ocr_fallback_rows(
    extracted: Dict[str, Any],
    fallback_rows: List[Dict[str, Any]],
    page_index_default: int,
) -> Dict[str, Any]:
    if not fallback_rows:
        return extracted

    line_items = _safe_list(extracted.get("line_items"))
    tables = _safe_list(extracted.get("tables"))
    detected_columns = set(_safe_list(extracted.get("detected_columns")))
    totals = extracted.get("totals") if isinstance(extracted.get("totals"), dict) else {"total_ht": [], "total_ttc": [], "tax": []}

    def _sig(item: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        return (
            str(item.get("reference") or ""),
            _compact_spaces(item.get("product")).upper(),
            str(item.get("quantity") or ""),
            str(item.get("unit_price") or ""),
            str(item.get("total") or ""),
            str(item.get("page_index") or ""),
        )

    seen = {_sig(item) for item in line_items if isinstance(item, dict)}
    next_table_index = 0
    for t in tables:
        if isinstance(t, dict):
            next_table_index = max(next_table_index, int(t.get("table_index") or 0))

    group_to_table: Dict[int, int] = {}
    grouped_rows: Dict[int, List[Dict[str, Any]]] = {}

    for row in fallback_rows:
        if not isinstance(row, dict):
            continue
        group = int(row.get("_group") or 1)
        if group not in group_to_table:
            next_table_index += 1
            group_to_table[group] = next_table_index
            grouped_rows[next_table_index] = []
        table_idx = group_to_table[group]

        item = {
            "table_index": table_idx,
            "page_index": page_index_default,
            "row_index": len(grouped_rows[table_idx]) + 1,
            "reference": _normalize_reference_code(row.get("reference")) or _extract_reference_code(
                " ".join(str(x) for x in _safe_list(row.get("raw_cells")))
            ),
            "product": row.get("product"),
            "quantity": row.get("quantity"),
            "unit_price": row.get("unit_price"),
            "total_ht": None,
            "total_ttc": None,
            "total": row.get("total"),
            "raw_cells": _safe_list(row.get("raw_cells")) or [str(row.get("product") or "")],
            "confidence": float(row.get("confidence") or 0.55),
        }
        sig = _sig(item)
        if sig in seen:
            continue
        seen.add(sig)
        line_items.append(item)
        grouped_rows[table_idx].append(item)
        if item.get("product"):
            detected_columns.add("product")
        if item.get("reference"):
            detected_columns.add("reference")
        if item.get("quantity"):
            detected_columns.add("quantity")
        if item.get("unit_price"):
            detected_columns.add("unit_price")
        if item.get("total"):
            detected_columns.add("total")

    for table_idx, rows in grouped_rows.items():
        if not rows:
            continue
        tables.append(
            {
                "table_index": table_idx,
                "page_index": page_index_default,
                "table_type": "line_items",
                "header_map": {"reference": 0, "product": 1, "quantity": 2, "unit_price": 3, "total": 4},
                "header_score": 0.0,
                "rows_count": len(rows),
                "shape": {
                    "source": "ocr-fallback",
                    "columns_estimated": 6,
                    "rows_estimated": int(len(rows)),
                },
                "rows": rows,
            }
        )

    return {
        "tables_count": len(tables),
        "rows_total": len(line_items),
        "detected_columns": sorted(detected_columns),
        "totals": totals,
        "tables": tables,
        "line_items": line_items,
    }


def _merge_ocr_totals_rows(
    extracted: Dict[str, Any],
    totals_rows: List[Dict[str, Any]],
    page_index_default: int,
) -> Dict[str, Any]:
    if not totals_rows:
        return extracted

    line_items = _safe_list(extracted.get("line_items"))
    tables = _safe_list(extracted.get("tables"))
    totals = extracted.get("totals") if isinstance(extracted.get("totals"), dict) else {}
    for key in ("total_ht", "total_ttc", "tax", "total", "amount_due", "stamp"):
        if key not in totals or not isinstance(totals.get(key), list):
            totals[key] = []
    for row in totals_rows:
        if not isinstance(row, dict):
            continue
        label = _compact_spaces(row.get("label"))
        value = _to_amount(row.get("value"))
        if not label or not value:
            continue
        field_key = str(row.get("field_key") or "total")
        if value not in totals.get(field_key, []):
            totals[field_key].append(value)

    return {
        "tables_count": len(tables),
        "rows_total": len(line_items),
        "detected_columns": _infer_detected_columns_from_rows(line_items),
        "totals": totals,
        "tables": tables,
        "line_items": line_items,
    }


def _extract_code_keys_from_text(value: Any) -> set[str]:
    out: set[str] = set()
    text = str(value or "")
    for m in re.finditer(r"(?i)(?:^|[^0-9,\\.])([A-Za-z]?\d{3,6})(?:$|[^0-9,\\.])", text):
        token = str(m.group(1) or "")
        digits = re.sub(r"\D", "", token)
        if not digits:
            continue
        key = digits.lstrip("0") or "0"
        out.add(key)
    return out


def _table_metrics(table: Dict[str, Any]) -> Dict[str, Any]:
    rows = [r for r in _safe_list(table.get("rows")) if isinstance(r, dict)]
    numeric_rows = 0
    code_keys: set[str] = set()
    for row in rows:
        if row.get("unit_price") or row.get("total") or row.get("total_ht") or row.get("total_ttc"):
            numeric_rows += 1
        code_keys.update(_extract_code_keys_from_text(row.get("reference")))
        code_keys.update(_extract_code_keys_from_text(row.get("product")))
        for raw in _safe_list(row.get("raw_cells")):
            code_keys.update(_extract_code_keys_from_text(raw))
    shape = table.get("shape") if isinstance(table.get("shape"), dict) else {}
    columns_estimated = int(shape.get("columns_estimated") or 0)
    return {
        "rows_count": len(rows),
        "numeric_rows": numeric_rows,
        "code_keys": code_keys,
        "columns_estimated": columns_estimated,
        "table_type": str(table.get("table_type") or ""),
    }


def _infer_detected_columns_from_rows(rows: List[Dict[str, Any]]) -> List[str]:
    cols = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("reference"):
            cols.add("reference")
        if row.get("product"):
            cols.add("product")
        if row.get("quantity"):
            cols.add("quantity")
        if row.get("unit_price"):
            cols.add("unit_price")
        if row.get("total") or row.get("total_ht") or row.get("total_ttc"):
            cols.add("total")
    return sorted(cols)


def _prune_redundant_tables(extracted: Dict[str, Any]) -> Dict[str, Any]:
    tables = [t for t in _safe_list(extracted.get("tables")) if isinstance(t, dict)]
    if len(tables) <= 1:
        return extracted

    metrics = {int(t.get("table_index") or i + 1): _table_metrics(t) for i, t in enumerate(tables)}
    rich_idxs = {
        idx
        for idx, m in metrics.items()
        if (m["numeric_rows"] >= 4)
        or (m["rows_count"] >= 6)
        or (m["columns_estimated"] >= 4 and m["numeric_rows"] >= 3)
    }
    if not rich_idxs:
        return extracted

    drop_idxs = set()
    for idx, m in metrics.items():
        if idx in rich_idxs:
            continue
        weak = (m["rows_count"] <= 4 and m["numeric_rows"] <= 1) or (m["table_type"] == "codes_only")
        if not weak:
            continue
        if any(metrics[r]["numeric_rows"] >= 4 and metrics[r]["rows_count"] >= 6 for r in rich_idxs):
            drop_idxs.add(idx)
            continue
        weak_codes = set(m["code_keys"])
        if not weak_codes:
            continue
        for rich_idx in rich_idxs:
            rich_codes = set(metrics[rich_idx]["code_keys"])
            if rich_codes and weak_codes.issubset(rich_codes):
                drop_idxs.add(idx)
                break
    if not drop_idxs:
        return extracted

    kept_tables = [t for t in tables if int(t.get("table_index") or 0) not in drop_idxs]
    for new_idx, table in enumerate(kept_tables, start=1):
        table["table_index"] = new_idx
        rows = [r for r in _safe_list(table.get("rows")) if isinstance(r, dict)]
        for row_idx, row in enumerate(rows, start=1):
            row["table_index"] = new_idx
            row["row_index"] = row_idx
        table["rows"] = rows
        table["rows_count"] = len(rows)

    rebuilt_rows: List[Dict[str, Any]] = []
    seen = set()
    for table in kept_tables:
        for row in _safe_list(table.get("rows")):
            if not isinstance(row, dict):
                continue
            sig = (
                str(row.get("page_index") or ""),
                str(row.get("reference") or ""),
                _compact_spaces(row.get("product")).upper(),
                str(row.get("quantity") or ""),
                str(row.get("unit_price") or ""),
                str(row.get("total_ht") or ""),
                str(row.get("total_ttc") or ""),
                str(row.get("total") or ""),
            )
            if sig in seen:
                continue
            seen.add(sig)
            rebuilt_rows.append(row)

    return {
        "tables_count": len(kept_tables),
        "rows_total": len(rebuilt_rows),
        "detected_columns": _infer_detected_columns_from_rows(rebuilt_rows),
        "totals": extracted.get("totals") if isinstance(extracted.get("totals"), dict) else {},
        "tables": kept_tables,
        "line_items": rebuilt_rows,
    }


def run_table_extraction(ctx: Dict[str, Any], profile: str) -> List[Dict[str, Any]]:
    docs = ctx.get("TOK_DOCS") or ctx.get("selected") or ctx.get("FINAL_DOCS") or []
    if not isinstance(docs, list):
        docs = []
    docs = _dedupe_docs(docs)
    source_map = _build_source_path_map(ctx)

    out: List[Dict[str, Any]] = []
    total_rows = 0
    total_tables = 0
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id")
        filename = str(doc.get("filename") or f"doc_{i}")
        extracted = _extract_doc_tables(doc, profile=profile)
        source_path = source_map.get(Path(filename).name.lower()) or source_map.get(filename.lower())
        if _should_run_ocr_fallback(extracted, source_path):
            fallback_pack = _ocr_fallback_tables_from_image(str(source_path))
            fallback_rows = _safe_list(fallback_pack.get("line_rows"))
            fallback_totals_rows = _safe_list(fallback_pack.get("totals_rows"))
            page_index_default = 1
            pages = _safe_list(doc.get("pages"))
            if pages and isinstance(pages[0], dict):
                page_index_default = int(pages[0].get("page_index") or pages[0].get("page") or 1)
            extracted = _merge_ocr_fallback_rows(extracted, fallback_rows, page_index_default)
            extracted = _merge_ocr_totals_rows(extracted, fallback_totals_rows, page_index_default)

        extracted = _filter_complete_line_items(extracted)
        extracted = _prune_redundant_tables(extracted)
        extracted = _filter_complete_line_items(extracted)

        total_rows += int(extracted.get("rows_total") or 0)
        total_tables += int(extracted.get("tables_count") or 0)

        out.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "engine": f"table-{profile}-unified-v6-anchor-geometry-line-items-only",
                "source_path": source_path,
                "tables_count": extracted.get("tables_count"),
                "rows_total": extracted.get("rows_total"),
                "detected_columns": extracted.get("detected_columns"),
                "totals": extracted.get("totals"),
                "line_items": extracted.get("line_items"),
                "tables": extracted.get("tables"),
            }
        )

    if profile == "50ml":
        ctx["TABLE_EXTRACTIONS_50ML"] = out
    elif profile == "100ml":
        ctx["TABLE_EXTRACTIONS_100ML"] = out
    else:
        ctx["TABLE_EXTRACTIONS_DEFAULT"] = out
    ctx["TABLE_EXTRACTIONS"] = out

    print(
        f"[table-extraction-{profile}] docs={len(out)} | tables={total_tables} | rows={total_rows}"
    )
    for row in out:
        if not isinstance(row, dict):
            continue
        print(
            f"  - {row.get('filename')} | tables={int(row.get('tables_count') or 0)} | "
            f"rows={int(row.get('rows_total') or 0)} | cols={row.get('detected_columns') or []}"
        )

    return out
