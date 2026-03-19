from __future__ import annotations

import math
import re
import runpy
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE_SCRIPT = Path(__file__).resolve().with_name("extraction-regles-yaml.py")
TOKEN_RE = re.compile(r"[A-Za-z0-9_\u00C0-\u024F\u0600-\u06FF]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [str(t).strip().lower() for t in TOKEN_RE.findall(str(text or "")) if str(t).strip()]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _doc_key(doc_id: Any, filename: Any) -> str:
    sid = str(doc_id or "").strip()
    if sid:
        return f"id:{sid}"
    sfn = str(filename or "").strip().lower()
    return f"fn:{sfn}"


def _iter_doc_chunks(doc: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    pages = doc.get("pages")
    if isinstance(pages, list) and pages:
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_index = _safe_int(page.get("page_index") or page.get("page"), 1)
            sents = page.get("sentences_layout")
            if isinstance(sents, list) and sents:
                for sent_index, sent in enumerate(sents):
                    if not isinstance(sent, dict):
                        continue
                    text = str(sent.get("text") or "")
                    if not text.strip():
                        continue
                    yield {
                        "page_index": page_index,
                        "sent_index": sent_index,
                        "text": text,
                        "tokens": _tokenize(text),
                    }
                continue

            page_text = str(page.get("page_text") or page.get("ocr_text") or page.get("text") or "")
            if page_text.strip():
                yield {
                    "page_index": page_index,
                    "sent_index": 0,
                    "text": page_text,
                    "tokens": _tokenize(page_text),
                }
        return

    text = str(doc.get("text") or "")
    if text.strip():
        yield {
            "page_index": 1,
            "sent_index": 0,
            "text": text,
            "tokens": _tokenize(text),
        }


def _build_docs_lookup(ctx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    docs = ctx.get("TOK_DOCS") or ctx.get("selected") or ctx.get("FINAL_DOCS") or []
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(docs, list):
        return out
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id")
        filename = doc.get("filename")
        out[_doc_key(doc_id, filename)] = doc
        if filename:
            out[f"fn:{str(filename).strip().lower()}"] = doc
        if doc_id:
            out[f"id:{str(doc_id).strip()}"] = doc
        out.setdefault(f"idx:{i}", doc)
    return out


def _collect_query_terms(fields: Dict[str, Any], doc_type: str) -> List[str]:
    terms: List[str] = []
    for field_name, field_cfg in (fields or {}).items():
        if isinstance(field_name, str):
            terms.extend(_tokenize(field_name.replace("_", " ")))
        matches = (field_cfg or {}).get("matches") if isinstance(field_cfg, dict) else []
        if not isinstance(matches, list):
            continue
        for m in matches:
            if not isinstance(m, dict):
                continue
            terms.extend(_tokenize(str(m.get("value") or "")))
    if not terms:
        terms.extend(_tokenize(str(doc_type or "").replace("_", " ")))

    out: List[str] = []
    seen = set()
    for t in terms:
        if len(t) < 2:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= 48:
            break
    return out


def _bm25_scores(
    docs_tokens: List[List[str]],
    query_terms: List[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    n_docs = len(docs_tokens)
    if n_docs == 0 or not query_terms:
        return [0.0 for _ in range(n_docs)]

    avgdl = sum(len(toks) for toks in docs_tokens) / float(max(1, n_docs))
    avgdl = max(avgdl, 1.0)

    df = Counter()
    for toks in docs_tokens:
        unique = set(toks)
        for term in query_terms:
            if term in unique:
                df[term] += 1

    idf: Dict[str, float] = {}
    for term in query_terms:
        n = float(df.get(term, 0))
        idf[term] = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))

    scores: List[float] = []
    for toks in docs_tokens:
        tf = Counter(toks)
        dl = float(max(1, len(toks)))
        score = 0.0
        for term in query_terms:
            freq = float(tf.get(term, 0))
            if freq <= 0.0:
                continue
            num = freq * (k1 + 1.0)
            den = freq + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf.get(term, 0.0) * (num / den if den > 0.0 else 0.0)
        scores.append(float(score))
    return scores


def _run_base_extraction(ctx: Dict[str, Any]) -> None:
    result = runpy.run_path(str(BASE_SCRIPT), run_name="__main__", init_globals=ctx)
    if isinstance(result, dict):
        ctx.update(result)


def _add_bm25(ctx: Dict[str, Any]) -> None:
    extractions = ctx.get("EXTRACTIONS")
    if not isinstance(extractions, list):
        return

    docs_lookup = _build_docs_lookup(ctx)
    bm25_results: List[Dict[str, Any]] = []

    for row in extractions:
        if not isinstance(row, dict):
            continue
        doc_id = row.get("doc_id")
        filename = row.get("filename")
        key = _doc_key(doc_id, filename)
        doc = docs_lookup.get(key)

        chunks = list(_iter_doc_chunks(doc or {}))
        docs_tokens = [c.get("tokens") or [] for c in chunks]
        query_terms = _collect_query_terms(row.get("fields") or {}, str(row.get("doc_type") or ""))
        scores = _bm25_scores(docs_tokens, query_terms, k1=1.5, b=0.75)

        ranked = []
        for i, chunk in enumerate(chunks):
            score = float(scores[i]) if i < len(scores) else 0.0
            ranked.append(
                {
                    "page_index": _safe_int(chunk.get("page_index"), 1),
                    "sent_index": _safe_int(chunk.get("sent_index"), 0),
                    "score": round(score, 6),
                    "text_preview": str(chunk.get("text") or "")[:280],
                }
            )
        ranked.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = ranked[:8]
        best_score = float(top_chunks[0]["score"]) if top_chunks else 0.0

        bm25 = {
            "k1": 1.5,
            "b": 0.75,
            "query_terms": query_terms,
            "chunks_total": len(chunks),
            "top_chunks": top_chunks,
            "best_score": round(best_score, 6),
            "avg_top3_score": round(
                sum(x["score"] for x in top_chunks[:3]) / float(max(1, len(top_chunks[:3]))),
                6,
            ),
        }
        row["bm25"] = bm25
        bm25_results.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "doc_type": row.get("doc_type"),
                "best_score": bm25["best_score"],
                "chunks_total": bm25["chunks_total"],
                "query_terms_count": len(query_terms),
            }
        )

    ctx["EXTRACTIONS"] = extractions
    ctx["BM25_RESULTS"] = bm25_results
    top_score = max((float(x.get("best_score") or 0.0) for x in bm25_results), default=0.0)
    print(
        "[extraction-regles-100ml] "
        f"docs={len(bm25_results)} | bm25_top_score={round(top_score, 6)}"
    )


_CTX = globals()
_run_base_extraction(_CTX)
_add_bm25(_CTX)
