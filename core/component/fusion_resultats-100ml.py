from __future__ import annotations

import json
from collections import defaultdict
import runpy
from pathlib import Path
from typing import Any, Dict, List


BASE_SCRIPT = Path(__file__).resolve().with_name("fusion_resultats.py")
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "fusion_output.json"


def _ml100_safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _ml100_doc_key(doc_id: Any, filename: Any) -> str:
    sid = str(doc_id or "").strip()
    if sid.lower() in {"non_specified", "none", "null", "na", "n/a"}:
        sid = ""
    if sid:
        return f"id:{sid}"
    sfn = str(filename or "").strip().lower()
    return f"fn:{sfn}"


def _ml100_index_rows(rows: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in _ml100_safe_list(rows):
        if not isinstance(row, dict):
            continue
        key = _ml100_doc_key(row.get("doc_id"), row.get("filename"))
        out[key] = row
        if row.get("filename"):
            raw_fn = str(row.get("filename")).strip().lower()
            out[f"fn:{raw_fn}"] = row
            out[f"fn:{Path(raw_fn).name}"] = row
        if row.get("doc_id"):
            out[f"id:{str(row.get('doc_id')).strip()}"] = row
    return out


def _ml100_group_rows(rows: Any) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in _ml100_safe_list(rows):
        if not isinstance(row, dict):
            continue
        keys = {_ml100_doc_key(row.get("doc_id"), row.get("filename"))}
        if row.get("filename"):
            raw_fn = str(row.get("filename")).strip().lower()
            keys.add(f"fn:{raw_fn}")
            keys.add(f"fn:{Path(raw_fn).name}")
        if row.get("doc_id"):
            keys.add(f"id:{str(row.get('doc_id')).strip()}")
        for key in keys:
            out.setdefault(key, []).append(row)
    return out


def _ml100_pick_bm25(extractions: Dict[str, Any], key: str) -> Dict[str, Any]:
    row = extractions.get(key)
    if not isinstance(row, dict):
        return {}
    bm25 = row.get("bm25")
    return bm25 if isinstance(bm25, dict) else {}


def _augment_payload(ctx: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    docs = _ml100_safe_list(payload.get("documents"))
    doc_vec_map = _ml100_index_rows(ctx.get("ML100_DOC_VECTORS"))
    topic_map = _ml100_index_rows(ctx.get("ML100_TOPICS"))
    chunk_map = _ml100_group_rows(ctx.get("ML100_CHUNK_VECTORS"))
    word_map = _ml100_group_rows(ctx.get("ML100_WORD_VECTORS"))
    ext_map = _ml100_index_rows(ctx.get("EXTRACTIONS"))

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("document_id")
        filename = (doc.get("file") or {}).get("name") if isinstance(doc.get("file"), dict) else None
        key = _ml100_doc_key(doc_id, filename)

        vec_row = doc_vec_map.get(key) or {}
        topic_row = topic_map.get(key) or {}
        doc_chunks = chunk_map.get(key) or []
        doc_words = word_map.get(key) or []
        bm25 = _ml100_pick_bm25(ext_map, key)
        doc_topics = _ml100_safe_list(topic_row.get("document_topics")) or _ml100_safe_list(topic_row.get("topics"))
        if not doc_topics:
            topic_scores: Dict[str, float] = defaultdict(float)
            for chunk in doc_chunks:
                if not isinstance(chunk, dict):
                    continue
                for item in _ml100_safe_list(chunk.get("chunk_topics")) or _ml100_safe_list(chunk.get("topics")):
                    if not isinstance(item, dict):
                        continue
                    term = str(item.get("term") or "").strip()
                    if not term:
                        continue
                    try:
                        score = float(item.get("score") or 0.0)
                    except Exception:
                        score = 0.0
                    topic_scores[term] += score
            doc_topics = [
                {"term": term, "score": round(score, 6)}
                for term, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:12]
            ]
        document_primary_topics = (
            _ml100_safe_list(topic_row.get("document_primary_topics"))[:2]
            or _ml100_safe_list(topic_row.get("main_topics"))[:2]
        )
        if not document_primary_topics:
            document_primary_topics = [
                str(item.get("term"))
                for item in doc_topics[:2]
                if isinstance(item, dict) and item.get("term")
            ]

        ml100_obj = {
            "embedding_method": ctx.get("ML100_EMBEDDING_METHOD"),
            "vector_dim": ctx.get("ML100_VECTOR_DIM"),
            "document_vector": vec_row.get("vector"),
            "chunk_count": int(vec_row.get("chunk_count") or len(doc_chunks)),
            "chunks_embeddings": doc_chunks,
            "word_embeddings": doc_words,
            "document_primary_topics": document_primary_topics,
            "document_topics": doc_topics,
        }
        doc["ml100"] = ml100_obj

        extraction = doc.get("extraction")
        if not isinstance(extraction, dict):
            extraction = {}
        if bm25:
            extraction["bm25"] = bm25
        doc["extraction"] = extraction

        components = doc.get("components")
        if not isinstance(components, dict):
            components = {}
        components["tokenisation_layout_100ml"] = {
            "embedding_method": ctx.get("ML100_EMBEDDING_METHOD"),
            "vector_dim": ctx.get("ML100_VECTOR_DIM"),
            "document_primary_topics": document_primary_topics,
            "document_topics_count": len(doc_topics),
            "chunk_vectors_count": len(doc_chunks),
            "word_vectors_count": len(doc_words),
        }
        components["extraction_regles_100ml"] = {
            "bm25_best_score": bm25.get("best_score") if isinstance(bm25, dict) else None,
            "bm25_chunks_total": bm25.get("chunks_total") if isinstance(bm25, dict) else None,
            "bm25_query_terms_count": len(_ml100_safe_list(bm25.get("query_terms"))) if isinstance(bm25, dict) else 0,
        }
        doc["components"] = components

        filename_label = str(filename or doc_id or "unknown")
        top_topics = [
            str(item.get("term"))
            for item in doc_topics[:5]
            if isinstance(item, dict) and item.get("term")
        ]
        print(
            f"[ml100-topic] {filename_label} | document_primary_topics={document_primary_topics} | "
            f"document_top_topics={top_topics}"
        )

    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, dict):
        pipeline = {}
    pipeline["profile"] = "pipeline100ml"
    pipeline["ml100"] = {
        "embedding_method": ctx.get("ML100_EMBEDDING_METHOD"),
        "vector_dim": ctx.get("ML100_VECTOR_DIM"),
        "doc_vectors_count": len(_ml100_safe_list(ctx.get("ML100_DOC_VECTORS"))),
        "chunk_vectors_count": len(_ml100_safe_list(ctx.get("ML100_CHUNK_VECTORS"))),
        "word_vectors_count": len(_ml100_safe_list(ctx.get("ML100_WORD_VECTORS"))),
        "topics_docs_count": len(_ml100_safe_list(ctx.get("ML100_TOPICS"))),
    }
    payload["pipeline"] = pipeline
    payload["documents"] = docs
    payload["documents_count"] = len(docs)
    return payload


def _run_base_fusion(ctx: Dict[str, Any]) -> None:
    result = runpy.run_path(str(BASE_SCRIPT), run_name="__main__", init_globals=ctx)
    if isinstance(result, dict):
        ctx.update(result)


def _load_payload(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run() -> None:
    ctx = globals()
    _run_base_fusion(ctx)

    out_path_raw = str(ctx.get("FUSION_RESULT") or OUTPUT_PATH)
    out_path = Path(out_path_raw)
    if not out_path.is_absolute():
        out_path = OUTPUT_PATH

    payload = _load_payload(out_path)
    if not payload:
        return

    payload = _augment_payload(ctx, payload)
    _save_payload(out_path, payload)

    ctx["FUSION_PAYLOAD"] = payload
    ctx["FUSION_PAYLOADS"] = payload.get("documents") if isinstance(payload.get("documents"), list) else []
    ctx["FUSION_RESULT"] = str(out_path)

    print(
        "[fusion-resultats-100ml] "
        f"docs={len(_ml100_safe_list(payload.get('documents')))} | "
        f"doc_vectors={len(_ml100_safe_list(ctx.get('ML100_DOC_VECTORS')))} | "
        f"bm25_docs={len(_ml100_safe_list(ctx.get('BM25_RESULTS')))}"
    )


_run()
