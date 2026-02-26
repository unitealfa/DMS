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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "fusion_output.json"


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


def build_payload(ctx: Dict[str, Any]) -> Dict[str, Any]:
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


def main() -> None:
    ctx = globals()
    payload = build_payload(ctx)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Impression et logging basique pour apparaître dans outputgeneralterminal.txt
    print("[Component: fusion-resultats]")
    print(f"[fusion-result] JSON fusionné écrit dans {OUTPUT_PATH}")
    print("[fusion-result] payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
