from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.elasticsearch import (  # noqa: E402
    fetch_sources_for_ids,
    index_tok_docs,
    maybe_build_store,
    to_classification_docs,
    to_extraction_docs,
)


def _normalize_ids(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for v in values:
        if v:
            out.append(str(v))
    return out


def run(ctx: Dict[str, Any]) -> Dict[str, Any]:
    store = maybe_build_store(ctx)
    if store is None:
        print("[elasticsearch] desactive ou indisponible -> fallback flux local.")
        ctx["ES_AVAILABLE"] = False
        ctx.setdefault("ES_DOC_IDS", [])
        ctx.setdefault("ES_CLASSIFICATION_DOCS", [])
        ctx.setdefault("ES_EXTRACTION_DOCS", [])
        return ctx

    ctx["ES_AVAILABLE"] = True
    store.ensure_index()

    base_docs = ctx.get("TOK_DOCS") or ctx.get("selected") or []
    es_doc_ids = index_tok_docs(store, base_docs)
    if not es_doc_ids:
        es_doc_ids = _normalize_ids(ctx.get("ES_DOC_IDS"))

    sources = fetch_sources_for_ids(store, es_doc_ids)
    cls_docs = to_classification_docs(sources)
    ext_docs = to_extraction_docs(sources)

    ctx["ES_DOC_IDS"] = es_doc_ids
    ctx["ES_CLASSIFICATION_DOCS"] = cls_docs
    ctx["ES_EXTRACTION_DOCS"] = ext_docs

    print(
        "[elasticsearch] index="
        f"{store.index} | docs_indexed={len(es_doc_ids)} | "
        f"classification_docs={len(cls_docs)} | extraction_docs={len(ext_docs)}"
    )
    return ctx


_CTX = globals()
run(_CTX)
