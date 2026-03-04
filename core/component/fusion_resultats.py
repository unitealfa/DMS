"""
Fusionne les sorties du pipeline en un JSON unique selon les directives de
"prompte .txt". Le script lit les variables globales (injectées par les
composants exécutés via runpy.run_path) et construit une structure complète
avec des valeurs par défaut prudentes.

Usage :
    python -m component.fusion_resultats
ou  runpy.run_path("component/fusion_resultats.py", init_globals=ctx)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "fusion_output.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.elasticsearch import fetch_sources_for_ids, maybe_build_store  # noqa: E402


# ---------- Helpers ----------
def ns() -> str:
    return "non_specified"


def first(lst: List[Any]) -> Any:
    return lst[0] if lst else None


def merge_list(*args: Optional[List[Any]]) -> List[Any]:
    out: List[Any] = []
    for a in args:
        if isinstance(a, list):
            out.extend(a)
    return out


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_load_json(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _build_map(rows: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_fn: Dict[str, Dict[str, Any]] = {}
    if not isinstance(rows, list):
        return {"by_id": by_id, "by_fn": by_fn}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("doc_id"):
            by_id[str(row["doc_id"])] = row
        if row.get("filename"):
            by_fn[str(row["filename"])] = row
    return {"by_id": by_id, "by_fn": by_fn}


def _pick_from_map(
    rows_map: Dict[str, Dict[str, Dict[str, Any]]],
    doc_id: Optional[str],
    filename: Optional[str],
) -> Dict[str, Any]:
    if doc_id and doc_id in rows_map["by_id"]:
        return rows_map["by_id"][doc_id]
    if filename and filename in rows_map["by_fn"]:
        return rows_map["by_fn"][filename]
    return {}


def extract_text_raw(ctx: Dict[str, Any]) -> str:
    # Try FINAL_DOCS then DOCS/TEXT_DOCS; fallback empty string.
    finals = ctx.get("FINAL_DOCS") or []
    if isinstance(finals, list) and finals:
        parts = []
        for d in finals:
            if isinstance(d, dict) and d.get("text"):
                parts.append(str(d.get("text")))
        if parts:
            return "\n\n".join(parts)
    docs = ctx.get("DOCS") or ctx.get("TEXT_DOCS") or []
    if isinstance(docs, list) and docs:
        parts = []
        for d in docs:
            if isinstance(d, dict):
                if d.get("text"):
                    parts.append(str(d["text"]))
                elif d.get("pages"):
                    for p in d["pages"]:
                        t = p.get("ocr_text") or p.get("text")
                        if t:
                            parts.append(str(t))
            elif isinstance(d, str):
                parts.append(d)
        if parts:
            return "\n\n".join(parts)
    return ""


def extract_pages_meta(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Placeholder: if PAGES_META exists, use it, else empty list.
    pm = ctx.get("PAGES_META")
    return pm if isinstance(pm, list) else []


def extract_pages(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Prefer TOK_DOCS or selected.
    return ctx.get("TOK_DOCS") or ctx.get("selected") or []


def extract_classification(ctx: Dict[str, Any]) -> Dict[str, Any]:
    res = ctx.get("RESULTS") or []
    cls = first(res) or {}
    return cls if isinstance(cls, dict) else {}


def extract_doc_type(cls: Dict[str, Any]) -> str:
    return str(cls.get("doc_type") or ns())


def extract_extractions(ctx: Dict[str, Any]) -> Any:
    return ctx.get("EXTRACTIONS") or []


def extract_detected_languages(ctx: Dict[str, Any]) -> List[str]:
    return ctx.get("DETECTED_LANGUAGES") or []


def build_payload_from_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    classification = extract_classification(ctx)
    doc_type = extract_doc_type(classification)
    text_raw = extract_text_raw(ctx)
    pages_meta = extract_pages_meta(ctx)
    pages = extract_pages(ctx)
    extractions = extract_extractions(ctx)
    detected_langs = extract_detected_languages(ctx)

    payload: Dict[str, Any] = {
        "document_id": ctx.get("DOC_ID") or ns(),
        "file": {
            "path": ctx.get("INPUT_FILE") or [],
            "name": None,
            "size": None,
        },
        "content": {
            "content_type": doc_type,
            "classification": classification or ns(),
            "document_kind": doc_type,
            "detected_languages": detected_langs,
        },
        "text": {
            "text_raw": text_raw,
            "text_normalized": ctx.get("TEXT_NORMALIZED") or "",
            "normalization": ctx.get("TEXT_NORMALIZATION") or [],
            "search": {
                "full_text": text_raw,
                "title": ctx.get("TITLE") or None,
                "keywords": ctx.get("SEARCH_KEYWORDS") or [],
            },
        },
        "document_structure": {
            "pages_meta": pages_meta,
            "pages": pages,
            "sections": ctx.get("SECTIONS") or [],
            "blocks": ctx.get("BLOCKS") or [],
            "lines": ctx.get("LINES") or [],
            "words": ctx.get("WORDS") or [],
            "headers": ctx.get("HEADERS") or [],
            "footers": ctx.get("FOOTERS") or [],
            "lists": ctx.get("LISTS") or [],
            "tables": ctx.get("TABLES") or [],
            "figures": ctx.get("FIGURES") or [],
            "equations": ctx.get("EQUATIONS") or [],
            "key_value_pairs": ctx.get("KEY_VALUE_PAIRS") or [],
            "reading_order": ctx.get("READING_ORDER") or [],
            "detected_columns": ctx.get("DETECTED_COLUMNS") or [],
            "non_text_regions": ctx.get("NON_TEXT_REGIONS") or [],
        },
        "ocr": {
            "orientation": ctx.get("OCR_ORIENTATION"),
            "confidence": ctx.get("OCR_CONFIDENCE"),
            "engine": ctx.get("OCR_ENGINE"),
        },
        "extraction": {
            "image_preprocessing": ctx.get("PRETRAITEMENT_RESULT") or {},
            "method": ctx.get("PREPROCESS_RESULT", {}).get("method") if isinstance(ctx.get("PREPROCESS_RESULT"), dict) else None,
            "native": ctx.get("PREPROCESS_RESULT", {}).get("NATIVE_FILES") if isinstance(ctx.get("PREPROCESS_RESULT"), dict) else None,
            "tesseract": ctx.get("PREPROCESS_RESULT", {}).get("IMAGE_ONLY_FILES") if isinstance(ctx.get("PREPROCESS_RESULT"), dict) else None,
            "regex_extractions": extractions,
            "business": ctx.get("BUSINESS") or {},
            "relations": ctx.get("RELATIONS") or [],
            "quality_checks": ctx.get("QUALITY_CHECKS") or [],
        },
        "nlp": {
            "language": ctx.get("NLP_LANGUAGE"),
            "sentences": ctx.get("NLP_SENTENCES") or [],
            "entities": ctx.get("NLP_ENTITIES") or [],
            "matches": ctx.get("NLP_MATCHES") or [],
            "pos": ctx.get("NLP_POS") or [],
            "lemma": ctx.get("NLP_LEMMA") or [],
        },
        "processing": {
            "warnings": ctx.get("PROCESS_WARNINGS") or [],
            "logs": ctx.get("PROCESS_LOGS") or [],
            "pipeline": ctx.get("PIPELINE_STEPS") or [],
            "durations": ctx.get("PROCESS_DURATIONS") or {},
            "timestamp": ctx.get("PROCESS_TIMESTAMP"),
        },
        "quality_checks": ctx.get("GLOBAL_QUALITY_CHECKS") or [],
        "human_review": {
            "required": bool(ctx.get("HUMAN_REVIEW_REQUIRED", False)),
            "tasks": ctx.get("HUMAN_REVIEW_TASKS") or [],
        },
    }
    return payload


def _es_text_from_pages(src: Dict[str, Any]) -> str:
    texts: List[str] = []
    for page in _safe_list(src.get("pages")):
        if not isinstance(page, dict):
            continue
        txt = str(page.get("text") or "")
        if txt.strip():
            texts.append(txt)
    return "\n\n".join(texts)


def _es_pages(src: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, page in enumerate(_safe_list(src.get("pages")), start=1):
        if not isinstance(page, dict):
            continue
        out.append(
            {
                "page_index": int(page.get("page_index") or i),
                "text": page.get("text") or "",
                "lang": page.get("lang"),
                "source_path": page.get("source_path") or "",
            }
        )
    return out


def build_payload_from_es_source(
    ctx: Dict[str, Any],
    src: Dict[str, Any],
    cls_map: Dict[str, Dict[str, Dict[str, Any]]],
    ext_map: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    doc_id = str(src.get("doc_id") or src.get("_id") or ns())
    filename = str(src.get("filename") or doc_id)
    paths = _safe_list(src.get("paths"))

    from_ctx_cls = _pick_from_map(cls_map, doc_id, filename)
    from_ctx_ext = _pick_from_map(ext_map, doc_id, filename)
    classification = from_ctx_cls if from_ctx_cls else (src.get("classification") or {})
    extraction = from_ctx_ext
    if not extraction:
        extraction = _safe_load_json(src.get("rule_extraction_payload")) or (src.get("rule_extraction") or {})

    doc_type = str(
        (classification.get("doc_type") if isinstance(classification, dict) else None)
        or src.get("doc_type")
        or ns()
    )

    text_raw = str(src.get("full_text") or "") or _es_text_from_pages(src)
    pages = _es_pages(src)
    words = _safe_list(src.get("words"))
    detected_langs = _safe_list(src.get("detected_languages"))

    payload: Dict[str, Any] = {
        "document_id": doc_id,
        "file": {
            "path": paths,
            "name": filename,
            "size": None,
        },
        "content": {
            "content_type": doc_type,
            "classification": classification or ns(),
            "document_kind": doc_type,
            "detected_languages": detected_langs,
        },
        "text": {
            "text_raw": text_raw,
            "text_normalized": ctx.get("TEXT_NORMALIZED") or "",
            "normalization": ctx.get("TEXT_NORMALIZATION") or [],
            "search": {
                "full_text": text_raw,
                "title": ctx.get("TITLE") or filename,
                "keywords": words[:256],
            },
        },
        "document_structure": {
            "pages_meta": extract_pages_meta(ctx),
            "pages": pages,
            "sections": ctx.get("SECTIONS") or [],
            "blocks": ctx.get("BLOCKS") or [],
            "lines": ctx.get("LINES") or [],
            "words": words,
            "headers": ctx.get("HEADERS") or [],
            "footers": ctx.get("FOOTERS") or [],
            "lists": ctx.get("LISTS") or [],
            "tables": ctx.get("TABLES") or [],
            "figures": ctx.get("FIGURES") or [],
            "equations": ctx.get("EQUATIONS") or [],
            "key_value_pairs": ctx.get("KEY_VALUE_PAIRS") or [],
            "reading_order": ctx.get("READING_ORDER") or [],
            "detected_columns": ctx.get("DETECTED_COLUMNS") or [],
            "non_text_regions": ctx.get("NON_TEXT_REGIONS") or [],
        },
        "ocr": {
            "orientation": ctx.get("OCR_ORIENTATION"),
            "confidence": ctx.get("OCR_CONFIDENCE"),
            "engine": ctx.get("OCR_ENGINE"),
        },
        "extraction": {
            "image_preprocessing": ctx.get("PRETRAITEMENT_RESULT") or {},
            "method": src.get("extraction"),
            "native": ctx.get("PREPROCESS_RESULT", {}).get("NATIVE_FILES") if isinstance(ctx.get("PREPROCESS_RESULT"), dict) else None,
            "tesseract": ctx.get("PREPROCESS_RESULT", {}).get("IMAGE_ONLY_FILES") if isinstance(ctx.get("PREPROCESS_RESULT"), dict) else None,
            "regex_extractions": extraction or [],
            "business": ctx.get("BUSINESS") or {},
            "relations": ctx.get("RELATIONS") or [],
            "quality_checks": ctx.get("QUALITY_CHECKS") or [],
        },
        "nlp": {
            "language": ctx.get("NLP_LANGUAGE"),
            "sentences": ctx.get("NLP_SENTENCES") or [],
            "entities": ctx.get("NLP_ENTITIES") or [],
            "matches": ctx.get("NLP_MATCHES") or [],
            "pos": ctx.get("NLP_POS") or [],
            "lemma": ctx.get("NLP_LEMMA") or [],
        },
        "processing": {
            "warnings": ctx.get("PROCESS_WARNINGS") or [],
            "logs": ctx.get("PROCESS_LOGS") or [],
            "pipeline": ctx.get("PIPELINE_STEPS") or [],
            "durations": ctx.get("PROCESS_DURATIONS") or {},
            "timestamp": ctx.get("PROCESS_TIMESTAMP"),
        },
        "quality_checks": ctx.get("GLOBAL_QUALITY_CHECKS") or [],
        "human_review": {
            "required": bool(ctx.get("HUMAN_REVIEW_REQUIRED", False)),
            "tasks": ctx.get("HUMAN_REVIEW_TASKS") or [],
        },
    }
    return payload


def build_payloads_from_es(ctx: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Composant debug: lit depuis ES pour construire un JSON de visualisation.
    Ne doit pas écrire dans ES ni bloquer le pipeline.
    """
    try:
        store = maybe_build_store(ctx)
        if store is None:
            return [], 0

        es_ids = [str(x) for x in _safe_list(ctx.get("ES_DOC_IDS")) if x]
        es_sources = fetch_sources_for_ids(store, es_ids)
        if not es_sources:
            return [], 0

        cls_map = _build_map(ctx.get("RESULTS"))
        ext_map = _build_map(ctx.get("EXTRACTIONS"))
        payloads = [build_payload_from_es_source(ctx, src, cls_map, ext_map) for src in es_sources if isinstance(src, dict)]
        return payloads, 0
    except Exception:
        return [], 0


def main() -> None:
    ctx = globals()
    payloads, es_synced = build_payloads_from_es(ctx)
    source = "elasticsearch"

    if payloads:
        payload: Any = payloads[0] if len(payloads) == 1 else {
            "documents": payloads,
            "count": len(payloads),
            "source": source,
        }
    else:
        source = "local-context"
        payload = build_payload_from_context(ctx)
        payloads = [payload]

    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx["FUSION_RESULT"] = str(OUTPUT_PATH)
    ctx["FUSION_PAYLOAD"] = payload
    ctx["FUSION_PAYLOADS"] = payloads
    ctx["FUSION_SOURCE"] = source
    ctx["ES_FUSION_SYNCED"] = es_synced

    # Impression et logging basique pour apparaître dans outputgeneralterminal.txt
    print("[Component: fusion-resultats]")
    print(f"[fusion-result] source={source} | docs={len(payloads)} | es_synced={es_synced}")
    print(f"[fusion-result] JSON fusionne ecrit dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
