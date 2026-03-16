from __future__ import annotations

import json
from collections import defaultdict
import runpy
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set


BASE_SCRIPT = Path(__file__).resolve().with_name("fusion_resultats.py")
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "fusion_output.json"


def _ml100_safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


_GRAMMAR_BLOCK_POS = {
    "PRON", "DET", "ADP", "CCONJ", "SCONJ", "CONJ", "PART", "AUX", "INTJ", "PUNCT", "SYM",
    "ADV", "RB", "RBR", "RBS",
    "PRP", "PRP$", "WP", "WP$", "WDT", "DT", "IN", "TO", "CC", "MD", "UH", "EX", "PDT", "POS", "RP",
}

_GRAMMAR_BLOCK_TERMS = {
    "_", "∅",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "moi", "te", "toi", "se", "lui", "leur", "leurs",
    "ce", "cet", "cette", "ces", "cela", "ca", "ça", "qui", "que", "quoi", "dont", "ou", "où",
    "de", "du", "des", "la", "le", "les", "un", "une", "et", "mais", "ou", "donc", "or", "ni", "car",
    "plus", "moins", "tres", "très", "avec", "pour", "par", "dans", "sur", "aux", "au",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "who", "which", "whom", "whose", "what", "where", "when", "why", "how",
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "هو", "هي", "هم", "هن", "انا", "أنت", "انت", "أنتم", "نحن", "ذلك", "هذه", "هذا", "الذي", "التي",
    "من", "في", "على", "الى", "إلى", "عن", "و", "او", "أو",
}


def _ml100_norm_term(value: Any) -> str:
    txt = str(value or "").strip().lower()
    if not txt:
        return ""
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = txt.replace("’", "'").replace("`", "'")
    return txt


def _ml100_doc_key(doc_id: Any, filename: Any) -> str:
    sid = str(doc_id or "").strip()
    if sid.lower() in {"non_specified", "none", "null", "na", "n/a"}:
        sid = ""
    if sid:
        return f"id:{sid}"
    sfn = str(filename or "").strip().lower()
    return f"fn:{sfn}"


def _ml100_doc_aliases(doc_id: Any, filename: Any) -> List[str]:
    aliases = {_ml100_doc_key(doc_id, filename)}
    sid = str(doc_id or "").strip()
    if sid:
        aliases.add(f"id:{sid}")
    sfn = str(filename or "").strip().lower()
    if sfn:
        aliases.add(f"fn:{sfn}")
        aliases.add(f"fn:{Path(sfn).name}")
    return [a for a in aliases if a]


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


def _ml100_is_grammar_noise(token_row: Dict[str, Any]) -> bool:
    pos = str(token_row.get("pos") or "").strip().upper()
    if pos in _GRAMMAR_BLOCK_POS:
        return True
    if any(pos.startswith(prefix) for prefix in ("PRON", "DET", "ADV", "CONJ", "AUX", "ADP")):
        return True

    token = _ml100_norm_term(token_row.get("token"))
    lemma = _ml100_norm_term(token_row.get("lemma"))
    if token in _GRAMMAR_BLOCK_TERMS or lemma in _GRAMMAR_BLOCK_TERMS:
        return True
    if token in {"_", "∅"} or lemma in {"_", "∅"}:
        return True
    return False


def _ml100_build_grammar_block_map(ctx: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for row in _ml100_safe_list(ctx.get("NLP_TOKENS")):
        if not isinstance(row, dict):
            continue
        if not _ml100_is_grammar_noise(row):
            continue
        token = _ml100_norm_term(row.get("token"))
        lemma = _ml100_norm_term(row.get("lemma"))
        aliases = _ml100_doc_aliases(row.get("doc_id"), row.get("filename") or row.get("doc"))
        for key in aliases:
            bucket = out.setdefault(key, set())
            if token:
                bucket.add(token)
            if lemma:
                bucket.add(lemma)
    return out


def _ml100_collect_blocked_terms(
    blocked_map: Dict[str, Set[str]],
    doc_id: Any,
    filename: Any,
) -> Set[str]:
    out: Set[str] = set()
    for key in _ml100_doc_aliases(doc_id, filename):
        values = blocked_map.get(key)
        if values:
            out.update(values)
    return out


def _ml100_is_blocked_topic_term(term: str, blocked_terms: Set[str]) -> bool:
    if not blocked_terms:
        return False
    norm = _ml100_norm_term(term)
    if not norm:
        return True
    if norm in blocked_terms:
        return True
    parts = [p for p in norm.split() if p]
    if not parts:
        return True
    if len(parts) == 1:
        return parts[0] in blocked_terms
    return any(p in blocked_terms for p in parts)


def _ml100_filter_topics(topics: Any, blocked_terms: Set[str]) -> tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    seen = set()
    removed = 0

    for item in _ml100_safe_list(topics):
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        if not term:
            continue
        norm = _ml100_norm_term(term)
        if not norm:
            removed += 1
            continue
        if norm in seen:
            continue
        if _ml100_is_blocked_topic_term(term, blocked_terms):
            removed += 1
            continue
        seen.add(norm)
        out.append(item)

    return out, removed


def _ml100_filter_chunk_topics(chunks: Any, blocked_terms: Set[str]) -> tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    removed = 0

    for chunk in _ml100_safe_list(chunks):
        if not isinstance(chunk, dict):
            continue
        row = dict(chunk)
        src_topics = _ml100_safe_list(row.get("chunk_topics")) or _ml100_safe_list(row.get("topics"))
        clean_topics, removed_here = _ml100_filter_topics(src_topics, blocked_terms)
        removed += removed_here
        row["chunk_topics"] = clean_topics
        row["chunk_primary_topic"] = clean_topics[0]["term"] if clean_topics else None
        out.append(row)

    return out, removed


def _augment_payload(ctx: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    docs = _ml100_safe_list(payload.get("documents"))
    doc_vec_map = _ml100_index_rows(ctx.get("ML100_DOC_VECTORS"))
    topic_map = _ml100_index_rows(ctx.get("ML100_TOPICS"))
    chunk_map = _ml100_group_rows(ctx.get("ML100_CHUNK_VECTORS"))
    word_map = _ml100_group_rows(ctx.get("ML100_WORD_VECTORS"))
    ext_map = _ml100_index_rows(ctx.get("EXTRACTIONS"))
    blocked_map = _ml100_build_grammar_block_map(ctx)

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("document_id")
        filename = (doc.get("file") or {}).get("name") if isinstance(doc.get("file"), dict) else None
        key = _ml100_doc_key(doc_id, filename)
        blocked_terms = _ml100_collect_blocked_terms(blocked_map, doc_id, filename)

        vec_row = doc_vec_map.get(key) or {}
        topic_row = topic_map.get(key) or {}
        raw_doc_chunks = chunk_map.get(key) or []
        doc_chunks, removed_chunk_topics = _ml100_filter_chunk_topics(raw_doc_chunks, blocked_terms)
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
        doc_topics, removed_doc_topics = _ml100_filter_topics(doc_topics, blocked_terms)
        document_primary_topics = (
            _ml100_safe_list(topic_row.get("document_primary_topics"))[:2]
            or _ml100_safe_list(topic_row.get("main_topics"))[:2]
        )
        document_primary_topics = [
            str(topic).strip()
            for topic in _ml100_safe_list(document_primary_topics)
            if str(topic).strip() and not _ml100_is_blocked_topic_term(str(topic), blocked_terms)
        ][:2]
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
            "topics_removed_by_grammar": int(removed_doc_topics + removed_chunk_topics),
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
            f"document_top_topics={top_topics} | blocked_terms={len(blocked_terms)} | "
            f"topics_removed={removed_doc_topics + removed_chunk_topics}"
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
