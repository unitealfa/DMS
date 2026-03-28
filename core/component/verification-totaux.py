from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
MONEY_TOLERANCE = Decimal("0.02")


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.translate(ARABIC_DIGITS)
    text = text.replace("\xa0", " ").replace("€", " ").replace("$", " ").replace("£", " ")
    text = text.replace("MAD", " ").replace("EUR", " ").replace("USD", " ")
    text = re.sub(r"[^0-9,.\-]", "", text)
    if not text:
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    if text.count(".") > 1:
        parts = text.split(".")
        text = "".join(parts[:-1]) + "." + parts[-1]

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _money_str(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _is_close(a: Optional[Decimal], b: Optional[Decimal], tolerance: Decimal = MONEY_TOLERANCE) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance


def _norm_match_text(value: Any) -> str:
    text = str(value or "").translate(ARABIC_DIGITS)
    text = text.replace("\xa0", " ").replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9A-Za-zÀ-ÿ%./:+\- ]+", " ", text)
    return " ".join(text.strip().lower().split())


def _doc_aliases(doc_id: Any, filename: Any, idx: int) -> List[str]:
    aliases = [_doc_key(doc_id, filename, idx)]
    sid = str(doc_id or "").strip()
    sfn = str(filename or "").strip().lower()
    if sid:
        aliases.append(f"id:{sid}")
    if sfn:
        aliases.append(f"fn:{sfn}")
        aliases.append(f"fn:{Path(sfn).name}")
    out: List[str] = []
    seen = set()
    for item in aliases:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _build_chunk_lookup(ctx: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    docs = ctx.get("TOK_DOCS") or ctx.get("selected") or []
    if not isinstance(docs, list):
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id")
        filename = doc.get("filename")
        rows: List[Dict[str, Any]] = []
        for page in _safe_list(doc.get("pages")):
            if not isinstance(page, dict):
                continue
            page_index = _safe_int(page.get("page_index"), 1)
            sent_items = _safe_list(page.get("sentences_layout") or page.get("sentences") or page.get("chunks"))
            for sent_index, item in enumerate(sent_items):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "")
                rows.append(
                    {
                        "page_index": page_index,
                        "sent_index": sent_index,
                        "text": text,
                        "text_norm": _norm_match_text(text),
                        "layout_kind": str(item.get("layout_kind") or ""),
                        "start": _safe_int(item.get("start"), 0),
                        "end": _safe_int(item.get("end"), 0),
                        "line": _safe_int(item.get("line"), 0),
                        "col": _safe_int(item.get("col"), 0),
                        "spans": item.get("spans") if isinstance(item.get("spans"), list) else [],
                        "table_rows": item.get("table_rows") if isinstance(item.get("table_rows"), list) else [],
                    }
                )
        for alias in _doc_aliases(doc_id, filename, idx):
            out[alias] = rows
    return out


def _value_position_in_text(text: str, candidate: str) -> Optional[Tuple[int, int]]:
    if not text or not candidate:
        return None
    pos = text.lower().find(candidate.lower())
    if pos >= 0:
        return pos, pos + len(candidate)
    return None


def _locate_chunk(
    chunk_lookup: Dict[str, List[Dict[str, Any]]],
    doc_id: Any,
    filename: Any,
    idx: int,
    page_index: Optional[int],
    candidates: List[str],
    prefer_layout: Optional[str] = None,
) -> Dict[str, Any]:
    chunks: List[Dict[str, Any]] = []
    for alias in _doc_aliases(doc_id, filename, idx):
        if alias in chunk_lookup:
            chunks = chunk_lookup[alias]
            break
    if not chunks:
        return {}

    norm_candidates = [_norm_match_text(x) for x in candidates if _norm_match_text(x)]
    best: Dict[str, Any] = {}
    best_score = -1.0
    for chunk in chunks:
        if page_index is not None and _safe_int(chunk.get("page_index"), 0) != int(page_index):
            continue
        text_norm = str(chunk.get("text_norm") or "")
        score = 0.0
        for cand in norm_candidates:
            if not cand:
                continue
            if cand == text_norm:
                score = max(score, 10.0 + len(cand) / 100.0)
            elif cand in text_norm:
                score = max(score, 7.0 + len(cand) / 100.0)
            else:
                parts = [p for p in cand.split() if len(p) >= 2]
                overlap = sum(1 for p in parts if p in text_norm)
                if parts:
                    score = max(score, overlap / float(len(parts)) * 4.0)
        if prefer_layout and str(chunk.get("layout_kind") or "") == prefer_layout:
            score += 0.75
        if score > best_score:
            best_score = score
            best = chunk

    if best_score <= 0.0 or not best:
        return {}

    source = {
        "page_index": _safe_int(best.get("page_index"), 0),
        "sent_index": _safe_int(best.get("sent_index"), 0),
        "layout_kind": best.get("layout_kind"),
        "line": _safe_int(best.get("line"), 0),
        "col": _safe_int(best.get("col"), 0),
        "chunk_start": _safe_int(best.get("start"), 0),
        "chunk_end": _safe_int(best.get("end"), 0),
        "chunk_text_excerpt": str(best.get("text") or "")[:400],
        "match_score": round(best_score, 6),
    }
    first_cand = next((c for c in candidates if str(c or "").strip()), "")
    pos = _value_position_in_text(str(best.get("text") or ""), str(first_cand or ""))
    if pos:
        source["match_start_in_chunk"] = pos[0]
        source["match_end_in_chunk"] = pos[1]
    return source


def _dominant_table_index(line_items: List[Dict[str, Any]]) -> Optional[int]:
    counts: Dict[int, int] = {}
    for row in line_items:
        table_index = _safe_int(row.get("table_index"), 0)
        if table_index <= 0:
            continue
        counts[table_index] = counts.get(table_index, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def _table_anchor_location(
    chunk_lookup: Dict[str, List[Dict[str, Any]]],
    doc_id: Any,
    filename: Any,
    idx: int,
    line_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    page_index = None
    dominant = _dominant_table_index(line_items)
    for row in line_items:
        if dominant is not None and _safe_int(row.get("table_index"), 0) == dominant:
            page_index = _safe_int(row.get("page_index"), 0) or None
            break
    row_candidates: List[str] = []
    for row in line_items[:3]:
        raw_cells = _safe_list(row.get("raw_cells"))
        if raw_cells:
            row_candidates.append(" ".join(str(x or "") for x in raw_cells))
        label = _row_label(row)
        total = row.get("total") or row.get("total_ht") or row.get("total_ttc")
        if label or total:
            row_candidates.append(f"{label} {total or ''}".strip())
    anchor = _locate_chunk(chunk_lookup, doc_id, filename, idx, page_index, row_candidates, prefer_layout="table")
    if dominant is not None:
        anchor["table_index"] = dominant
    return anchor


def _row_total_candidate(row: Dict[str, Any]) -> Optional[Decimal]:
    for key in ("total", "total_ht", "total_ttc"):
        parsed = _to_decimal(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _row_label(row: Dict[str, Any]) -> str:
    for key in ("product", "reference"):
        txt = str(row.get(key) or "").strip()
        if txt:
            return txt[:120]
    return f"row#{_safe_int(row.get('row_index'), 0)}"


def _candidate_values(values: Any) -> List[Decimal]:
    out: List[Decimal] = []
    seen = set()
    for raw in _safe_list(values):
        parsed = _to_decimal(raw)
        if parsed is None:
            continue
        key = str(parsed.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        if key in seen:
            continue
        seen.add(key)
        out.append(parsed)
    return out


def _pick_nearest(target: Optional[Decimal], values: List[Decimal]) -> Optional[Decimal]:
    if target is None or not values:
        return None
    return min(values, key=lambda x: (abs(x - target), x))


def _pick_total_candidate(
    subtotal: Optional[Decimal],
    tax: Optional[Decimal],
    total_candidates: List[Decimal],
) -> Optional[Decimal]:
    if not total_candidates:
        return None
    if subtotal is not None and tax is not None:
        expected = subtotal + tax
        return _pick_nearest(expected, total_candidates)
    if subtotal is not None:
        larger = [v for v in total_candidates if v >= subtotal]
        if larger:
            return _pick_nearest(subtotal, larger)
        return _pick_nearest(subtotal, total_candidates)
    return max(total_candidates)


def _pick_subtotal_candidate(subtotal: Optional[Decimal], totals: Dict[str, List[Decimal]]) -> Optional[Decimal]:
    candidates = list(totals.get("total_ht") or [])
    if not candidates:
        candidates = list(totals.get("total") or [])
    return _pick_nearest(subtotal, candidates) if candidates else None


def _pick_tax_candidate(
    subtotal: Optional[Decimal],
    total_value: Optional[Decimal],
    tax_candidates: List[Decimal],
) -> Optional[Decimal]:
    if not tax_candidates:
        return None
    if subtotal is not None and total_value is not None:
        expected = total_value - subtotal
        return _pick_nearest(expected, tax_candidates)
    return max(tax_candidates)


def _verify_row(
    row: Dict[str, Any],
    chunk_lookup: Dict[str, List[Dict[str, Any]]],
    doc_id: Any,
    filename: Any,
    idx: int,
) -> Dict[str, Any]:
    qty = _to_decimal(row.get("quantity"))
    unit_price = _to_decimal(row.get("unit_price"))
    declared_total = _row_total_candidate(row)
    computed_total = None
    if qty is not None and unit_price is not None:
        computed_total = (qty * unit_price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    if computed_total is not None and declared_total is not None:
        status = "ok" if _is_close(computed_total, declared_total) else "mismatch"
    elif computed_total is not None or declared_total is not None:
        status = "partial"
    else:
        status = "missing"

    effective_total = None
    effective_source = None
    if computed_total is not None and declared_total is not None:
        if _is_close(computed_total, declared_total):
            effective_total = declared_total
            effective_source = "declared+computed"
        else:
            effective_total = computed_total
            effective_source = "computed"
    elif computed_total is not None:
        effective_total = computed_total
        effective_source = "computed"
    elif declared_total is not None:
        effective_total = declared_total
        effective_source = "declared"

    difference = None
    if computed_total is not None and declared_total is not None:
        difference = declared_total - computed_total

    raw_cells = _safe_list(row.get("raw_cells"))
    source_location = _locate_chunk(
        chunk_lookup,
        doc_id,
        filename,
        idx,
        _safe_int(row.get("page_index"), 0) or None,
        [
            " ".join(str(x or "") for x in raw_cells),
            _row_label(row),
            f"{_row_label(row)} {row.get('total') or row.get('total_ht') or row.get('total_ttc') or ''}".strip(),
            str(row.get("total") or row.get("total_ht") or row.get("total_ttc") or ""),
        ],
        prefer_layout="table",
    )

    return {
        "page_index": _safe_int(row.get("page_index"), 0),
        "table_index": _safe_int(row.get("table_index"), 0),
        "row_index": _safe_int(row.get("row_index"), 0),
        "label": _row_label(row),
        "raw_cells": raw_cells,
        "quantity": _money_str(qty),
        "unit_price": _money_str(unit_price),
        "declared_total": _money_str(declared_total),
        "computed_total": _money_str(computed_total),
        "effective_total": _money_str(effective_total),
        "effective_total_source": effective_source,
        "status": status,
        "difference": _money_str(difference),
        "source_location": source_location,
    }


def _doc_key(doc_id: Any, filename: Any, idx: int) -> str:
    sid = str(doc_id or "").strip()
    if sid:
        return f"id:{sid}"
    sfn = str(filename or "").strip().lower()
    if sfn:
        return f"fn:{sfn}"
    return f"idx:{idx}"


def _locate_total_value(
    chunk_lookup: Dict[str, List[Dict[str, Any]]],
    doc_id: Any,
    filename: Any,
    idx: int,
    line_items: List[Dict[str, Any]],
    field_name: str,
    value: Optional[Decimal],
    extra_labels: List[str],
) -> Dict[str, Any]:
    page_index = None
    dominant = _dominant_table_index(line_items)
    for row in line_items:
        if dominant is not None and _safe_int(row.get("table_index"), 0) == dominant:
            page_index = _safe_int(row.get("page_index"), 0) or None
            break
    value_text = _money_str(value) or ""
    candidates = [f"{label} {value_text}".strip() for label in extra_labels if label or value_text]
    if value_text:
        candidates.append(value_text)
    located = _locate_chunk(chunk_lookup, doc_id, filename, idx, page_index, candidates, prefer_layout="table")
    if dominant is not None:
        located["table_index"] = dominant
    if field_name:
        located["field"] = field_name
    return located


def _verify_doc(row: Dict[str, Any], idx: int, chunk_lookup: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    doc_id = row.get("doc_id")
    filename = str(row.get("filename") or f"doc_{idx}")
    totals_raw = row.get("totals") if isinstance(row.get("totals"), dict) else {}
    line_items = [item for item in _safe_list(row.get("line_items")) if isinstance(item, dict)]
    table_anchor = _table_anchor_location(chunk_lookup, doc_id, filename, idx, line_items)

    row_audit = [_verify_row(item, chunk_lookup, doc_id, filename, idx) for item in line_items]
    rows_with_total = [r for r in row_audit if r.get("effective_total") is not None]

    computed_subtotal: Optional[Decimal] = None
    if rows_with_total:
        computed_subtotal = sum((Decimal(str(r["effective_total"])) for r in rows_with_total), Decimal("0.00"))

    parsed_totals = {
        "total_ht": _candidate_values(totals_raw.get("total_ht")),
        "total_ttc": _candidate_values(totals_raw.get("total_ttc")),
        "total": _candidate_values(totals_raw.get("total")),
        "amount_due": _candidate_values(totals_raw.get("amount_due")),
        "tax": _candidate_values(totals_raw.get("tax")),
    }

    declared_subtotal = _pick_subtotal_candidate(computed_subtotal, parsed_totals)
    total_candidates = list(parsed_totals["total_ttc"]) + list(parsed_totals["amount_due"]) + list(parsed_totals["total"])
    declared_total = _pick_total_candidate(declared_subtotal or computed_subtotal, None, total_candidates)
    declared_tax = _pick_tax_candidate(declared_subtotal or computed_subtotal, declared_total, parsed_totals["tax"])

    expected_total = None
    if (declared_subtotal or computed_subtotal) is not None and declared_tax is not None:
        expected_total = (declared_subtotal or computed_subtotal) + declared_tax
        if declared_total is None and total_candidates:
            declared_total = _pick_total_candidate(declared_subtotal or computed_subtotal, declared_tax, total_candidates)

    row_mismatch_count = sum(1 for r in row_audit if r.get("status") == "mismatch")
    row_ok_count = sum(1 for r in row_audit if r.get("status") == "ok")
    row_partial_count = sum(1 for r in row_audit if r.get("status") == "partial")

    subtotal_status = "missing"
    if computed_subtotal is not None and declared_subtotal is not None:
        subtotal_status = "ok" if _is_close(computed_subtotal, declared_subtotal) else "mismatch"
    elif computed_subtotal is not None or declared_subtotal is not None:
        subtotal_status = "partial"

    tax_status = "missing"
    computed_tax = None
    if declared_total is not None and (declared_subtotal or computed_subtotal) is not None:
        computed_tax = declared_total - (declared_subtotal or computed_subtotal)
    if computed_tax is not None and declared_tax is not None:
        tax_status = "ok" if _is_close(computed_tax, declared_tax) else "mismatch"
    elif declared_tax is not None:
        tax_status = "partial"

    total_status = "missing"
    if expected_total is not None and declared_total is not None:
        total_status = "ok" if _is_close(expected_total, declared_total) else "mismatch"
    elif expected_total is not None or declared_total is not None:
        total_status = "partial"

    checks = [
        {"name": "rows", "status": "ok" if row_mismatch_count == 0 and rows_with_total else ("mismatch" if row_mismatch_count else "partial")},
        {"name": "subtotal", "status": subtotal_status},
        {"name": "tax", "status": tax_status},
        {"name": "total", "status": total_status},
    ]
    available_checks = [c for c in checks if c["status"] != "missing"]
    mismatch_count = sum(1 for c in available_checks if c["status"] == "mismatch")
    passed = mismatch_count == 0 and (row_mismatch_count == 0)
    complete = all(c["status"] == "ok" for c in checks)

    if not available_checks and not row_audit:
        verification_status = "not_enough_data"
    elif mismatch_count > 0 or row_mismatch_count > 0:
        verification_status = "mismatch"
    elif complete:
        verification_status = "ok"
    else:
        verification_status = "partial_ok"

    subtotal_location = _locate_total_value(
        chunk_lookup, doc_id, filename, idx, line_items, "subtotal", declared_subtotal, ["total ht", "montant ht", "subtotal"]
    )
    tax_location = _locate_total_value(
        chunk_lookup, doc_id, filename, idx, line_items, "tax", declared_tax, ["tva", "vat", "tax", "taxe"]
    )
    total_location = _locate_total_value(
        chunk_lookup, doc_id, filename, idx, line_items, "total", declared_total, ["total", "total ttc", "amount due", "montant a payer"]
    )

    issue_locations: List[Dict[str, Any]] = []
    for audit_row in row_audit:
        if audit_row.get("status") != "mismatch":
            continue
        issue_locations.append(
            {
                "kind": "row_mismatch",
                "page_index": audit_row.get("page_index"),
                "table_index": audit_row.get("table_index"),
                "row_index": audit_row.get("row_index"),
                "label": audit_row.get("label"),
                "difference": audit_row.get("difference"),
                "source_location": audit_row.get("source_location"),
            }
        )
    if subtotal_status == "mismatch":
        issue_locations.append(
            {
                "kind": "subtotal_mismatch",
                "table_index": table_anchor.get("table_index"),
                "page_index": subtotal_location.get("page_index") or table_anchor.get("page_index"),
                "computed": _money_str(computed_subtotal),
                "declared": _money_str(declared_subtotal),
                "source_location": subtotal_location or table_anchor,
            }
        )
    elif subtotal_status == "partial":
        issue_locations.append(
            {
                "kind": "subtotal_partial",
                "table_index": table_anchor.get("table_index"),
                "page_index": subtotal_location.get("page_index") or table_anchor.get("page_index"),
                "computed": _money_str(computed_subtotal),
                "declared": _money_str(declared_subtotal),
                "source_location": subtotal_location or table_anchor,
            }
        )
    if tax_status == "mismatch":
        issue_locations.append(
            {
                "kind": "tax_mismatch",
                "table_index": table_anchor.get("table_index"),
                "page_index": tax_location.get("page_index") or table_anchor.get("page_index"),
                "computed": _money_str(computed_tax),
                "declared": _money_str(declared_tax),
                "source_location": tax_location or table_anchor,
            }
        )
    elif tax_status == "partial":
        issue_locations.append(
            {
                "kind": "tax_partial",
                "table_index": table_anchor.get("table_index"),
                "page_index": tax_location.get("page_index") or table_anchor.get("page_index"),
                "computed": _money_str(computed_tax),
                "declared": _money_str(declared_tax),
                "source_location": tax_location or table_anchor,
            }
        )
    if total_status == "mismatch":
        issue_locations.append(
            {
                "kind": "total_mismatch",
                "table_index": table_anchor.get("table_index"),
                "page_index": total_location.get("page_index") or table_anchor.get("page_index"),
                "computed": _money_str(expected_total),
                "declared": _money_str(declared_total),
                "source_location": total_location or table_anchor,
            }
        )
    elif total_status == "partial":
        issue_locations.append(
            {
                "kind": "total_partial",
                "table_index": table_anchor.get("table_index"),
                "page_index": total_location.get("page_index") or table_anchor.get("page_index"),
                "computed": _money_str(expected_total),
                "declared": _money_str(declared_total),
                "source_location": total_location or table_anchor,
            }
        )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "doc_key": _doc_key(doc_id, filename, idx),
        "engine": "totals-verification-0ml-v1",
        "tables_count": _safe_int(row.get("tables_count"), 0),
        "rows_total": _safe_int(row.get("rows_total"), 0),
        "rows_verified": len(rows_with_total),
        "row_ok_count": row_ok_count,
        "row_partial_count": row_partial_count,
        "row_mismatch_count": row_mismatch_count,
        "declared_totals_raw": totals_raw,
        "table_anchor": table_anchor,
        "computed_subtotal": _money_str(computed_subtotal),
        "declared_subtotal": _money_str(declared_subtotal),
        "declared_tax": _money_str(declared_tax),
        "computed_tax": _money_str(computed_tax),
        "declared_total": _money_str(declared_total),
        "expected_total": _money_str(expected_total),
        "subtotal_location": subtotal_location,
        "tax_location": tax_location,
        "total_location": total_location,
        "subtotal_status": subtotal_status,
        "tax_status": tax_status,
        "total_status": total_status,
        "passed": bool(passed),
        "complete": bool(complete),
        "verification_status": verification_status,
        "checks": checks,
        "issue_locations": issue_locations[:100],
        "row_audit": row_audit[:200],
        "tolerance": _money_str(MONEY_TOLERANCE),
    }


def run(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = ctx.get("TABLE_EXTRACTIONS")
    if not isinstance(rows, list):
        rows = (
            ctx.get("TABLE_EXTRACTIONS_DEFAULT")
            or ctx.get("TABLE_EXTRACTIONS_50ML")
            or ctx.get("TABLE_EXTRACTIONS_100ML")
            or []
        )
    if not isinstance(rows, list):
        rows = []

    chunk_lookup = _build_chunk_lookup(ctx)
    out = [_verify_doc(row, idx, chunk_lookup) for idx, row in enumerate(rows) if isinstance(row, dict)]
    ctx["TOTALS_VERIFICATION"] = out

    ok = sum(1 for row in out if str(row.get("verification_status")) == "ok")
    partial_ok = sum(1 for row in out if str(row.get("verification_status")) == "partial_ok")
    mismatch = sum(1 for row in out if str(row.get("verification_status")) == "mismatch")
    missing = sum(1 for row in out if str(row.get("verification_status")) == "not_enough_data")
    print(
        "[verification-totaux] "
        f"docs={len(out)} | ok={ok} | partial_ok={partial_ok} | mismatch={mismatch} | missing={missing}"
    )
    for row in out:
        print(
            "  - "
            f"{row.get('filename')} | status={row.get('verification_status')} | "
            f"rows={_safe_int(row.get('rows_verified'), 0)}/{_safe_int(row.get('rows_total'), 0)} | "
            f"subtotal={row.get('computed_subtotal')}~{row.get('declared_subtotal')} | "
            f"tax={row.get('declared_tax')} | total={row.get('declared_total')}"
        )
        first_issue = _safe_list(row.get("issue_locations"))[:1]
        if first_issue and isinstance(first_issue[0], dict):
            issue = first_issue[0]
            src = issue.get("source_location") if isinstance(issue.get("source_location"), dict) else {}
            print(
                "    -> "
                f"issue={issue.get('kind')} | table={issue.get('table_index') or src.get('table_index')} | "
                f"page={issue.get('page_index') or src.get('page_index')} | "
                f"chunk={src.get('sent_index')} | line={src.get('line')} | "
                f"start={src.get('chunk_start')} | end={src.get('chunk_end')}"
            )
    return out


_CTX = globals()
TOTALS_VERIFICATION = run(_CTX)
