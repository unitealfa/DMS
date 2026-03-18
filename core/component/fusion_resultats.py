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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _interdoc_aliases(doc_id: Optional[str], filename: Optional[str]) -> List[str]:
    aliases: List[str] = []
    sid = str(doc_id or "").strip()
    sfn = str(filename or "").strip()
    if sid:
        aliases.append(f"id:{sid}")
    if sfn:
        aliases.append(f"fn:{sfn}")
        aliases.append(f"fn:{Path(sfn).name}")
    if not aliases:
        return []
    # dedupe en conservant l'ordre
    out: List[str] = []
    seen = set()
    for key in aliases:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _interdoc_link_ids_for_doc(ctx: Dict[str, Any], doc_id: Optional[str], filename: Optional[str]) -> List[str]:
    mapping = ctx.get("INTERDOC_DOC_LINKS")
    if not isinstance(mapping, dict):
        return []
    ids: List[str] = []
    seen = set()
    for alias in _interdoc_aliases(doc_id, filename):
        rows = mapping.get(alias)
        if not isinstance(rows, list):
            continue
        for value in rows:
            sval = str(value or "").strip()
            if not sval or sval in seen:
                continue
            seen.add(sval)
            ids.append(sval)
    return ids


def _interdoc_output(ctx: Dict[str, Any]) -> Dict[str, Any]:
    analysis = ctx.get("INTERDOC_ANALYSIS")
    if not isinstance(analysis, dict):
        return {
            "method": None,
            "documents_analyzed": 0,
            "pairs_evaluated": 0,
            "sentence_pairs_scored": 0,
            "links_count": 0,
            "links": [],
            "generated_at": None,
        }
    return {
        "method": analysis.get("method"),
        "documents_analyzed": _safe_int(analysis.get("documents_analyzed"), 0),
        "pairs_evaluated": _safe_int(analysis.get("pairs_evaluated"), 0),
        "sentence_pairs_scored": _safe_int(analysis.get("sentence_pairs_scored"), 0),
        "links_count": _safe_int(analysis.get("links_count"), 0),
        "links": _safe_list(analysis.get("links")),
        "generated_at": analysis.get("generated_at"),
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _basename(value: Any) -> str:
    if value is None:
        return ""
    try:
        return Path(str(value)).name
    except Exception:
        return str(value)


def _norm_filename(value: Any) -> str:
    return _basename(value).strip().lower()


def _same_filename(a: Any, b: Any) -> bool:
    na = _norm_filename(a)
    nb = _norm_filename(b)
    return bool(na and nb and na == nb)


def _safe_non_negative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < 0:
        return None
    return parsed


def _size_from_paths(paths: List[Any]) -> Optional[int]:
    uniq: List[str] = []
    seen = set()
    for path in paths:
        sp = str(path or "").strip()
        if not sp or sp in seen:
            continue
        seen.add(sp)
        uniq.append(sp)
    if not uniq:
        return None
    total = 0
    found = False
    for sp in uniq:
        try:
            total += int(Path(sp).stat().st_size)
            found = True
        except Exception:
            continue
    if not found:
        return None
    return total


def _size_maps_from_pretraitement(ctx: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
    cached = ctx.get("_FUSION_PRETRAIT_SIZE_CACHE")
    if isinstance(cached, dict):
        by_path = cached.get("by_path")
        by_name = cached.get("by_name")
        if isinstance(by_path, dict) and isinstance(by_name, dict):
            return by_path, by_name

    by_path: Dict[str, int] = {}
    by_name: Dict[str, int] = {}
    for row in _safe_list(ctx.get("PRETRAITEMENT_RESULT")):
        if not isinstance(row, dict):
            continue
        size = _safe_non_negative_int(row.get("size"))
        if size is None:
            continue
        path = str(row.get("path") or "").strip()
        if path:
            by_path[path] = size
            name = _basename(path)
            if name and name not in by_name:
                by_name[name] = size
        else:
            name = str(row.get("filename") or "").strip()
            if name and name not in by_name:
                by_name[name] = size

    ctx["_FUSION_PRETRAIT_SIZE_CACHE"] = {"by_path": by_path, "by_name": by_name}
    return by_path, by_name


def _resolve_file_size(
    ctx: Dict[str, Any],
    paths: List[Any],
    filename: Optional[str],
    preferred: Any = None,
) -> Optional[int]:
    preferred_size = _safe_non_negative_int(preferred)
    if preferred_size is not None:
        return preferred_size

    by_path, by_name = _size_maps_from_pretraitement(ctx)
    for path in paths:
        sp = str(path or "").strip()
        if not sp:
            continue
        if sp in by_path:
            return by_path[sp]

    name = str(filename or "").strip()
    if name and name in by_name:
        return by_name[name]

    return _safe_non_negative_int(_size_from_paths(paths))


def _row_belongs_to_doc(row: Any, doc_id: Optional[str], filename: Optional[str]) -> bool:
    if not isinstance(row, dict):
        return False
    row_doc_id = str(row.get("doc_id") or row.get("document_id") or "").strip()
    row_filename = str(row.get("filename") or row.get("doc") or "").strip()
    if doc_id and row_doc_id and row_doc_id == doc_id:
        return True
    if filename and row_filename and (row_filename == filename or _same_filename(row_filename, filename)):
        return True
    return False


def _filter_rows_for_doc(rows: Any, doc_id: Optional[str], filename: Optional[str]) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if _row_belongs_to_doc(row, doc_id, filename):
            out.append(row)
    return out


def _doc_key(row: Dict[str, Any], index: int) -> str:
    doc_id = str(row.get("doc_id") or "").strip()
    if doc_id:
        return f"id:{doc_id}"
    filename = str(row.get("filename") or "").strip()
    if filename:
        return f"fn:{filename}"
    paths = _safe_list(row.get("paths"))
    if paths:
        return f"path:{_basename(paths[0])}"
    return f"idx:{index}"


def _doc_text_score(row: Dict[str, Any]) -> int:
    total = 0
    for page in _safe_list(row.get("pages")):
        if not isinstance(page, dict):
            continue
        txt = page.get("page_text") or page.get("text") or page.get("ocr_text") or ""
        if not txt and isinstance(page.get("sentences_layout"), list):
            txt = "\n".join(
                str(s.get("text") or "") if isinstance(s, dict) else str(s)
                for s in (page.get("sentences_layout") or [])
            )
        total += len(str(txt).strip())
    if total > 0:
        return total
    return len(str(row.get("text") or "").strip())


def _dedupe_docs(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    best: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        key = _doc_key(row, i)
        if key not in best:
            best[key] = row
            order.append(key)
            continue

        current = best[key]
        cur_score = _doc_text_score(current)
        new_score = _doc_text_score(row)
        cur_pages = len(_safe_list(current.get("pages")))
        new_pages = len(_safe_list(row.get("pages")))
        cur_content = str(current.get("content") or "").strip().lower()
        new_content = str(row.get("content") or "").strip().lower()

        replace = False
        if cur_content == "image_only" and new_content != "image_only":
            replace = True
        elif cur_content != "image_only" and new_content == "image_only":
            replace = False
        elif new_pages > cur_pages and new_content != "image_only":
            replace = True
        elif new_score > cur_score:
            replace = True
        elif new_score == cur_score and new_pages > cur_pages:
            replace = True

        if replace:
            best[key] = row

    return [best[k] for k in order]


def _normalize_pages_from_doc(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, page in enumerate(_safe_list(row.get("pages")), start=1):
        if not isinstance(page, dict):
            continue
        item = dict(page)
        if item.get("page_index") is None:
            item["page_index"] = i
        if "text" not in item:
            item["text"] = page.get("page_text") or page.get("ocr_text") or ""
        out.append(item)
    return out


def _filter_and_dedupe_extractions(
    rows: Any,
    doc_id: Optional[str],
    filename: Optional[str],
) -> List[Dict[str, Any]]:
    if isinstance(rows, dict):
        raw = [rows]
    elif isinstance(rows, list):
        raw = [r for r in rows if isinstance(r, dict)]
    else:
        return []

    filtered: List[Dict[str, Any]] = []
    for row in raw:
        row_doc_id = str(row.get("doc_id") or "").strip()
        row_filename = str(row.get("filename") or "").strip()
        if doc_id and row_doc_id and row_doc_id == doc_id:
            filtered.append(row)
            continue
        if filename and row_filename and row_filename == filename:
            filtered.append(row)

    if not filtered:
        filtered = raw

    best: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for i, row in enumerate(filtered):
        key = str(row.get("doc_id") or "").strip() or str(row.get("filename") or "").strip() or f"row#{i}"
        fields = row.get("fields") if isinstance(row.get("fields"), dict) else {}
        score = len(fields)
        if key not in best:
            best[key] = row
            order.append(key)
            continue
        cur_fields = best[key].get("fields") if isinstance(best[key].get("fields"), dict) else {}
        if score > len(cur_fields):
            best[key] = row
    return [best[k] for k in order]


def _default_nlp_tokens_index(ctx: Dict[str, Any]) -> str:
    idx = str(ctx.get("ES_NLP_INDEX_EFFECTIVE") or ctx.get("ES_NLP_INDEX") or "").strip()
    if idx:
        return idx
    base = str(ctx.get("ES_INDEX") or "dms_documents").strip() or "dms_documents"
    return f"{base}_nlp_tokens"


def _search_index(
    store: Any,
    index: str,
    query: Dict[str, Any],
    size: int,
    sort: Optional[List[Any]] = None,
    source_includes: Optional[List[str]] = None,
    search_after: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"size": size, "query": query}
    if sort:
        payload["sort"] = sort
    if source_includes:
        payload["_source"] = source_includes
    if search_after:
        payload["search_after"] = search_after
    path = f"/{quote(index)}/_search"
    return store._request("POST", path, payload=payload) or {}


def _fetch_nlp_tokens(
    store: Any,
    index: str,
    query: Dict[str, Any],
    max_tokens: int = 50000,
) -> Dict[str, Any]:
    sort = [
        {"page_index": "asc"},
        {"sent_index": "asc"},
        {"tok_index": "asc"},
        {"_id": "asc"},
    ]
    source_includes = [
        "doc_id",
        "filename",
        "lang",
        "page_index",
        "sent_index",
        "tok_index",
        "token",
        "lemma",
        "pos",
        "ner",
        "sentence_text",
        "updated_at",
    ]

    rows: List[Dict[str, Any]] = []
    search_after: Optional[List[Any]] = None
    total = 0
    batch_size = 2000

    while len(rows) < max_tokens:
        wanted = min(batch_size, max_tokens - len(rows))
        res = _search_index(
            store=store,
            index=index,
            query=query,
            size=wanted,
            sort=sort,
            source_includes=source_includes,
            search_after=search_after,
        )
        hits_root = (res.get("hits") or {}) if isinstance(res, dict) else {}
        total_info = hits_root.get("total")
        if isinstance(total_info, dict):
            total = _safe_int(total_info.get("value"), total)
        elif isinstance(total_info, int):
            total = total_info

        hits = hits_root.get("hits") or []
        if not isinstance(hits, list) or not hits:
            break
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            src = hit.get("_source")
            if isinstance(src, dict):
                rows.append(src)
            if len(rows) >= max_tokens:
                break

        last_sort = hits[-1].get("sort") if isinstance(hits[-1], dict) else None
        if not isinstance(last_sort, list):
            break
        search_after = last_sort

    if total <= 0:
        total = len(rows)
    return {
        "rows": rows,
        "total": total,
        "truncated": total > len(rows),
        "index": index,
    }


def _structure_nlp_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pages_map: Dict[int, Dict[str, Any]] = {}
    for row in tokens:
        if not isinstance(row, dict):
            continue
        page_index = _safe_int(row.get("page_index"), 0)
        sent_index = _safe_int(row.get("sent_index"), 0)
        page_entry = pages_map.setdefault(
            page_index,
            {"page_index": page_index, "_sent_map": {}},
        )
        sent_map = page_entry["_sent_map"]
        sent_entry = sent_map.setdefault(
            sent_index,
            {
                "sent_index": sent_index,
                "lang": row.get("lang"),
                "text": row.get("sentence_text"),
                "tokens": [],
            },
        )
        sent_entry["tokens"].append(
            {
                "tok_index": _safe_int(row.get("tok_index"), 0),
                "token": row.get("token"),
                "lemma": row.get("lemma"),
                "pos": row.get("pos"),
                "ner": row.get("ner"),
            }
        )

    pages_out: List[Dict[str, Any]] = []
    for page_index in sorted(pages_map.keys()):
        page = pages_map[page_index]
        sent_map = page.pop("_sent_map", {})
        sentences: List[Dict[str, Any]] = []
        for sent_index in sorted(sent_map.keys()):
            sent = sent_map[sent_index]
            sent["tokens"] = sorted(
                sent.get("tokens") or [],
                key=lambda x: _safe_int(x.get("tok_index"), 0),
            )
            sentences.append(sent)
        page["sentences"] = sentences
        pages_out.append(page)
    return pages_out


def _nlp_from_es_and_ctx(
    ctx: Dict[str, Any],
    src: Dict[str, Any],
    store: Any,
    doc_id: str,
    filename: str,
) -> Dict[str, Any]:
    es_nlp = src.get("nlp") if isinstance(src.get("nlp"), dict) else {}
    level = str(es_nlp.get("level") or ctx.get("ES_NLP_LEVEL") or "").strip().lower()
    if level not in {"off", "summary", "full"}:
        level = "summary"

    sentences = _filter_rows_for_doc(ctx.get("NLP_SENTENCES"), doc_id, filename)
    entities = _filter_rows_for_doc(ctx.get("NLP_ENTITIES"), doc_id, filename)
    tokens_rows = _filter_rows_for_doc(ctx.get("NLP_TOKENS"), doc_id, filename)
    if not tokens_rows:
        # Fallback compat si des runs anciens n'ont que NLP_POS/NLP_LEMMA.
        pos_rows = _filter_rows_for_doc(ctx.get("NLP_POS"), doc_id, filename)
        lemma_rows = _filter_rows_for_doc(ctx.get("NLP_LEMMA"), doc_id, filename)
        merged: Dict[Tuple[Any, Any, Any, Any, Any, Any], Dict[str, Any]] = {}
        for row in pos_rows:
            if not isinstance(row, dict):
                continue
            key = (
                row.get("filename"),
                row.get("page_index"),
                row.get("sent_index"),
                row.get("tok_index"),
                row.get("token"),
                row.get("lang"),
            )
            merged[key] = {
                "filename": row.get("filename"),
                "page_index": row.get("page_index"),
                "sent_index": row.get("sent_index"),
                "tok_index": row.get("tok_index"),
                "token": row.get("token"),
                "pos": row.get("pos"),
                "lemma": None,
                "ner": row.get("ner"),
                "lang": row.get("lang"),
            }
        for row in lemma_rows:
            if not isinstance(row, dict):
                continue
            key = (
                row.get("filename"),
                row.get("page_index"),
                row.get("sent_index"),
                row.get("tok_index"),
                row.get("token"),
                row.get("lang"),
            )
            cur = merged.get(key)
            if cur is None:
                merged[key] = {
                    "filename": row.get("filename"),
                    "page_index": row.get("page_index"),
                    "sent_index": row.get("sent_index"),
                    "tok_index": row.get("tok_index"),
                    "token": row.get("token"),
                    "pos": None,
                    "lemma": row.get("lemma"),
                    "ner": row.get("ner"),
                    "lang": row.get("lang"),
                }
            else:
                cur["lemma"] = row.get("lemma")
                if cur.get("ner") is None:
                    cur["ner"] = row.get("ner")
        tokens_rows = sorted(
            merged.values(),
            key=lambda x: (
                _safe_int(x.get("page_index"), 0),
                _safe_int(x.get("sent_index"), 0),
                _safe_int(x.get("tok_index"), 0),
            ),
        )

    out: Dict[str, Any] = {
        "source": "elasticsearch",
        "level": level,
        "language": ctx.get("NLP_LANGUAGE"),
        "summary": {
            "languages": _safe_list(es_nlp.get("languages")) or _safe_list(src.get("detected_languages")),
            "language_stats": es_nlp.get("language_stats") if isinstance(es_nlp, dict) else {},
            "sentences_count": _safe_int(es_nlp.get("sentences_count"), len(sentences)),
            "tokens_count": _safe_int(es_nlp.get("tokens_count"), len(tokens_rows)),
            "entities_count": _safe_int(es_nlp.get("entities_count"), len(entities)),
            "top_pos": _safe_list(es_nlp.get("top_pos")),
            "top_ner": _safe_list(es_nlp.get("top_ner")),
            "entities_sample": _safe_list(es_nlp.get("entities_sample_flat")) or _safe_list(es_nlp.get("entities_sample")),
            "sentences_sample": _safe_list(es_nlp.get("sentences_sample")),
            "updated_at": src.get("nlp_updated_at"),
        },
        "sentences": sentences,
        "entities": entities,
        "matches": _filter_rows_for_doc(ctx.get("NLP_MATCHES"), doc_id, filename),
        "tokens": tokens_rows,
    }

    if level == "full":
        tokens_index = _default_nlp_tokens_index(ctx)
        max_tokens = _safe_int(ctx.get("FUSION_NLP_TOKENS_MAX"), 50000)
        if max_tokens <= 0:
            max_tokens = 50000
        try:
            fetched = _fetch_nlp_tokens(
                store=store,
                index=tokens_index,
                query={"term": {"doc_id": doc_id}},
                max_tokens=max_tokens,
            )

            flat_rows = fetched["rows"]
            selected_doc_id = doc_id

            if not flat_rows and filename:
                fetched_by_name = _fetch_nlp_tokens(
                    store=store,
                    index=tokens_index,
                    query={"term": {"filename": filename}},
                    max_tokens=max_tokens,
                )
                rows_by_doc: Dict[str, List[Dict[str, Any]]] = {}
                for row in fetched_by_name["rows"]:
                    if not isinstance(row, dict):
                        continue
                    rid = str(row.get("doc_id") or "").strip() or "_missing_doc_id"
                    rows_by_doc.setdefault(rid, []).append(row)

                if rows_by_doc:
                    expected = _safe_int((es_nlp.get("tokens_count") if isinstance(es_nlp, dict) else None), 0)
                    best_id = None
                    best_rows: List[Dict[str, Any]] = []
                    best_score: Optional[Tuple[int, int]] = None
                    for rid, rid_rows in rows_by_doc.items():
                        size = len(rid_rows)
                        if expected > 0:
                            score = (abs(size - expected), -size)
                        else:
                            score = (0, -size)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_id = rid
                            best_rows = rid_rows
                    if best_id is not None:
                        selected_doc_id = best_id
                        flat_rows = best_rows
                        fetched["total"] = len(best_rows)
                        fetched["truncated"] = False

            out["full"] = {
                "index": fetched["index"],
                "count": fetched["total"],
                "returned": len(flat_rows),
                "truncated": bool(fetched["truncated"]),
                "doc_id": selected_doc_id,
                "tokens": flat_rows,
                "structure": _structure_nlp_tokens(flat_rows),
            }
        except Exception as exc:
            out["full"] = {
                "index": tokens_index,
                "count": 0,
                "returned": 0,
                "truncated": False,
                "doc_id": doc_id,
                "tokens": [],
                "structure": [],
                "error": str(exc),
            }

    return out


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


def _same_doc_hint(value: Any, filename: str, first_path: str) -> bool:
    if value is None:
        return False
    txt = str(value)
    if first_path and txt == first_path:
        return True
    return bool(filename and (txt == filename or _same_filename(txt, filename)))


def _extract_component_views(
    ctx: Dict[str, Any],
    doc_payload: Dict[str, Any],
) -> Dict[str, Any]:
    document_id = str(doc_payload.get("document_id") or "")
    file_obj = doc_payload.get("file") if isinstance(doc_payload.get("file"), dict) else {}
    filename = str(file_obj.get("name") or "")
    interdoc_link_ids = _interdoc_link_ids_for_doc(ctx, document_id, filename)
    interdoc = _interdoc_output(ctx)
    first_path = ""
    paths = _safe_list(file_obj.get("path"))
    if paths:
        first_path = str(paths[0])

    pretrait_rows = _safe_list(ctx.get("PRETRAITEMENT_RESULT"))
    pretrait_match = {}
    for row in pretrait_rows:
        if not isinstance(row, dict):
            continue
        if _same_doc_hint(row.get("path"), filename, first_path):
            pretrait_match = row
            break

    preprocess = ctx.get("PREPROCESS_RESULT") if isinstance(ctx.get("PREPROCESS_RESULT"), dict) else {}
    text_files = _safe_list(preprocess.get("TEXT_FILES"))
    image_files = _safe_list(preprocess.get("IMAGE_ONLY_FILES"))
    pre_docs = _safe_list(preprocess.get("DOCS"))
    is_text_file = any(_same_doc_hint(p, filename, first_path) for p in text_files)
    is_image_file = any(_same_doc_hint(p, filename, first_path) for p in image_files)

    final_rows = _safe_list(ctx.get("FINAL_DOCS"))
    final_doc = {}
    for row in final_rows:
        if not isinstance(row, dict):
            continue
        if _same_doc_hint(row.get("filename"), filename, first_path):
            final_doc = row
            break

    tok_rows = _safe_list(ctx.get("TOK_DOCS") or ctx.get("selected"))
    tok_doc = {}
    for row in tok_rows:
        if not isinstance(row, dict):
            continue
        if document_id and str(row.get("doc_id") or "") == document_id:
            tok_doc = row
            break
        if _same_doc_hint(row.get("filename"), filename, first_path):
            tok_doc = row
            break

    tok_pages = _safe_list(tok_doc.get("pages"))
    tok_sentences = 0
    for page in tok_pages:
        if not isinstance(page, dict):
            continue
        sent_items = page.get("sentences_layout") or page.get("sentences") or page.get("chunks") or []
        if isinstance(sent_items, list):
            tok_sentences += len(sent_items)

    nlp_obj = doc_payload.get("nlp") if isinstance(doc_payload.get("nlp"), dict) else {}
    nlp_summary = nlp_obj.get("summary") if isinstance(nlp_obj.get("summary"), dict) else {}
    cls_obj = doc_payload.get("content", {}).get("classification") if isinstance(doc_payload.get("content"), dict) else {}
    ext_obj = doc_payload.get("extraction") if isinstance(doc_payload.get("extraction"), dict) else {}
    regex_obj = ext_obj.get("regex_extractions")
    if isinstance(regex_obj, dict):
        regex_doc_count = 1
        regex_fields_count = len(regex_obj.get("fields") or {}) if isinstance(regex_obj.get("fields"), dict) else 0
    elif isinstance(regex_obj, list):
        regex_doc_count = len(regex_obj)
        regex_fields_count = sum(
            len((r.get("fields") or {})) for r in regex_obj if isinstance(r, dict) and isinstance(r.get("fields"), dict)
        )
    else:
        regex_doc_count = 0
        regex_fields_count = 0

    final_pages = _safe_int(final_doc.get("page_count_total"), 0)
    if final_pages <= 0:
        final_pages = len(_safe_list(final_doc.get("pages_text")))

    return {
        "pretraitement_de_docs": {
            "matched_path": pretrait_match.get("path"),
            "detected_ext": pretrait_match.get("ext"),
            "detected_mime": pretrait_match.get("mime"),
            "detected_label": pretrait_match.get("label"),
            "detected_content": pretrait_match.get("content"),
            "total_files_detected": len(pretrait_rows),
        },
        "si_image_pretraiter_sinonpass_le_doc": {
            "is_text_file": is_text_file,
            "is_image_only_file": is_image_file,
            "text_files_count": len(text_files),
            "image_only_files_count": len(image_files),
            "docs_prepared_count": len(pre_docs),
        },
        "output_txt": {
            "content": final_doc.get("content"),
            "extraction": final_doc.get("extraction"),
            "pages_count": final_pages,
            "has_text": bool(str((doc_payload.get("text") or {}).get("text_raw") or "").strip()),
            "text_length": len(str((doc_payload.get("text") or {}).get("text_raw") or "")),
        },
        "tokenisation_layout": {
            "pages_count": len(tok_pages),
            "sentences_count": tok_sentences,
            "detected_languages": _safe_list(tok_doc.get("detected_languages")),
        },
        "attribution_grammaticale": {
            "component": ctx.get("NLP_GRAMMAR_COMPONENT_NAME") or "atripusion-gramatical",
            "script": ctx.get("NLP_GRAMMAR_COMPONENT_SCRIPT"),
            "backend": ctx.get("NLP_GRAMMAR_BACKEND"),
            "model": ctx.get("NLP_GRAMMAR_MODEL"),
            "model_source": ctx.get("NLP_GRAMMAR_MODEL_SOURCE"),
            "model_install": ctx.get("NLP_GRAMMAR_MODEL_INSTALL"),
            "pos_method": ctx.get("NLP_POS_METHOD"),
            "pos_refined_count": _safe_int(ctx.get("NLP_POS_REFINED_COUNT"), 0),
            "pos_total": _safe_int(ctx.get("NLP_POS_TOTAL"), 0),
            "pos_refined_rate": float(ctx.get("NLP_POS_REFINED_RATE") or 0.0),
            "pos_top_tags": _safe_list(ctx.get("NLP_POS_TOP")),
            "level": nlp_obj.get("level"),
            "language": nlp_obj.get("language"),
            "languages": _safe_list(nlp_summary.get("languages")),
            "sentences_count": _safe_int(nlp_summary.get("sentences_count"), len(_safe_list(nlp_obj.get("sentences")))),
            "tokens_count": _safe_int(nlp_summary.get("tokens_count"), len(_safe_list((nlp_obj.get("full") or {}).get("tokens")))),
            "entities_count": _safe_int(nlp_summary.get("entities_count"), len(_safe_list(nlp_obj.get("entities")))),
        },
        "liaison_inter_docs": {
            "method": interdoc.get("method"),
            "links_count_total": _safe_int(interdoc.get("links_count"), 0),
            "linked_documents_count": len(interdoc_link_ids),
        },
        "elasticsearch": {
            "enabled": bool(ctx.get("USE_ELASTICSEARCH")),
            "available": bool(ctx.get("ES_AVAILABLE")),
            "es_url": ctx.get("ES_URL"),
            "es_index": ctx.get("ES_INDEX"),
            "doc_id": document_id or None,
            "nlp_level": ctx.get("ES_NLP_LEVEL_EFFECTIVE") or ctx.get("ES_NLP_LEVEL"),
            "nlp_tokens_index": ctx.get("ES_NLP_INDEX_EFFECTIVE") or ctx.get("ES_NLP_INDEX"),
            "nlp_docs_synced": _safe_int(ctx.get("ES_NLP_DOCS_SYNCED"), 0),
            "nlp_tokens_synced": _safe_int(ctx.get("ES_NLP_TOKENS_SYNCED"), 0),
            "nlp_token_errors": _safe_int(ctx.get("ES_NLP_TOKEN_ERRORS"), 0),
        },
        "classification": {
            "doc_type": cls_obj.get("doc_type") if isinstance(cls_obj, dict) else None,
            "status": cls_obj.get("status") if isinstance(cls_obj, dict) else None,
            "winning_score": cls_obj.get("winning_score") if isinstance(cls_obj, dict) else None,
        },
        "extraction_regles": {
            "documents_count": regex_doc_count,
            "fields_count": regex_fields_count,
        },
        "fusion_resultats": {
            "source": ctx.get("FUSION_SOURCE"),
            "output_path": str(OUTPUT_PATH),
            "es_synced": _safe_int(ctx.get("ES_FUSION_SYNCED"), 0),
        },
    }


def _to_document_output(
    ctx: Dict[str, Any],
    doc_payload: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:
    file_obj = doc_payload.get("file") if isinstance(doc_payload.get("file"), dict) else {}
    content_obj = doc_payload.get("content") if isinstance(doc_payload.get("content"), dict) else {}
    nlp_obj = doc_payload.get("nlp") if isinstance(doc_payload.get("nlp"), dict) else {}
    extraction_obj = doc_payload.get("extraction") if isinstance(doc_payload.get("extraction"), dict) else {}
    text_obj = doc_payload.get("text") if isinstance(doc_payload.get("text"), dict) else {}
    structure_obj = doc_payload.get("document_structure") if isinstance(doc_payload.get("document_structure"), dict) else {}
    document_id = str(doc_payload.get("document_id") or "")
    filename = str(file_obj.get("name") or "")
    interdoc_link_ids = _interdoc_link_ids_for_doc(ctx, document_id, filename)

    components = _extract_component_views(ctx, doc_payload)
    page_count = len(_safe_list(structure_obj.get("pages")))

    return {
        "document_id": document_id or doc_payload.get("document_id"),
        "file": {
            "name": file_obj.get("name"),
            "paths": _safe_list(file_obj.get("path")),
            "size": file_obj.get("size"),
            "page_count": page_count,
        },
        "classification": content_obj.get("classification"),
        "doc_type": content_obj.get("document_kind") or content_obj.get("content_type"),
        "cross_document": {
            "linked_documents_count": len(interdoc_link_ids),
            "link_ids": interdoc_link_ids,
        },
        "components": components,
        "text": text_obj,
        "document_structure": structure_obj,
        "ocr": doc_payload.get("ocr"),
        "extraction": extraction_obj,
        "nlp": nlp_obj,
        "quality_checks": doc_payload.get("quality_checks"),
        "human_review": doc_payload.get("human_review"),
        "processing": doc_payload.get("processing"),
        "meta": {
            "generated_at": _iso_now(),
            "schema_version": "2.0",
            "source": source,
        },
    }


def _final_output(
    ctx: Dict[str, Any],
    payloads: List[Dict[str, Any]],
    source: str,
) -> Dict[str, Any]:
    docs = [_to_document_output(ctx, p, source) for p in payloads if isinstance(p, dict)]
    interdoc = _interdoc_output(ctx)
    out: Dict[str, Any] = {
        "schema_version": "2.0",
        "generated_at": _iso_now(),
        "source": source,
        "documents_count": len(docs),
        "documents": docs,
        "cross_document_analysis": interdoc,
        "pipeline": {
            "es_enabled": bool(ctx.get("USE_ELASTICSEARCH")),
            "es_available": bool(ctx.get("ES_AVAILABLE")),
            "es_url": ctx.get("ES_URL"),
            "es_index": ctx.get("ES_INDEX"),
            "es_nlp_level": ctx.get("ES_NLP_LEVEL_EFFECTIVE") or ctx.get("ES_NLP_LEVEL"),
            "es_nlp_index": ctx.get("ES_NLP_INDEX_EFFECTIVE") or ctx.get("ES_NLP_INDEX"),
            "es_doc_ids": _safe_list(ctx.get("ES_DOC_IDS")),
            "steps": _safe_list(ctx.get("PIPELINE_STEPS")),
            "durations": ctx.get("PROCESS_DURATIONS") or {},
        },
    }
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
    tok_docs = _dedupe_docs(_safe_list(ctx.get("TOK_DOCS") or ctx.get("selected")))
    if tok_docs:
        pages = _normalize_pages_from_doc(tok_docs[0])
        if pages:
            return pages

    finals = _safe_list(ctx.get("FINAL_DOCS"))
    if finals and isinstance(finals[0], dict):
        first_final = finals[0]
        pages_text = _safe_list(first_final.get("pages_text"))
        if pages_text:
            source_path = str(first_final.get("source_path") or "")
            return [
                {
                    "page_index": i,
                    "text": str(txt or ""),
                    "source_path": source_path,
                }
                for i, txt in enumerate(pages_text, start=1)
            ]
        text = str(first_final.get("text") or "")
        if text.strip():
            return [{"page_index": 1, "text": text, "source_path": str(first_final.get("source_path") or "")}]
    return []


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


def _text_from_tok_pages(pages: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        txt = str(page.get("text") or page.get("page_text") or page.get("ocr_text") or "")
        if txt.strip():
            chunks.append(txt)
    return "\n\n".join(chunks)


def _pages_from_final_doc(final_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages_text = _safe_list(final_doc.get("pages_text"))
    if not pages_text:
        text = str(final_doc.get("text") or "")
        if not text.strip():
            return []
        pages_text = [text]
    source_path = str(final_doc.get("source_path") or "")
    return [
        {"page_index": i, "text": str(txt or ""), "source_path": source_path}
        for i, txt in enumerate(pages_text, start=1)
    ]


def _collect_local_tokens_for_doc(ctx: Dict[str, Any], doc_id: Optional[str], filename: Optional[str]) -> List[Dict[str, Any]]:
    tokens_rows = _filter_rows_for_doc(ctx.get("NLP_TOKENS"), doc_id, filename)
    if tokens_rows:
        return tokens_rows

    # Fallback compat runs anciens: merge NLP_POS + NLP_LEMMA pour CE document.
    pos_rows = _filter_rows_for_doc(ctx.get("NLP_POS"), doc_id, filename)
    lemma_rows = _filter_rows_for_doc(ctx.get("NLP_LEMMA"), doc_id, filename)
    merged: Dict[Tuple[Any, Any, Any, Any, Any, Any], Dict[str, Any]] = {}
    for row in pos_rows:
        if not isinstance(row, dict):
            continue
        key = (
            row.get("filename"),
            row.get("page_index"),
            row.get("sent_index"),
            row.get("tok_index"),
            row.get("token"),
            row.get("lang"),
        )
        merged[key] = {
            "filename": row.get("filename"),
            "page_index": row.get("page_index"),
            "sent_index": row.get("sent_index"),
            "tok_index": row.get("tok_index"),
            "token": row.get("token"),
            "pos": row.get("pos"),
            "lemma": None,
            "ner": row.get("ner"),
            "lang": row.get("lang"),
        }
    for row in lemma_rows:
        if not isinstance(row, dict):
            continue
        key = (
            row.get("filename"),
            row.get("page_index"),
            row.get("sent_index"),
            row.get("tok_index"),
            row.get("token"),
            row.get("lang"),
        )
        cur = merged.get(key)
        if cur is None:
            merged[key] = {
                "filename": row.get("filename"),
                "page_index": row.get("page_index"),
                "sent_index": row.get("sent_index"),
                "tok_index": row.get("tok_index"),
                "token": row.get("token"),
                "pos": None,
                "lemma": row.get("lemma"),
                "ner": row.get("ner"),
                "lang": row.get("lang"),
            }
        else:
            cur["lemma"] = row.get("lemma")
            if cur.get("ner") is None:
                cur["ner"] = row.get("ner")

    return sorted(
        merged.values(),
        key=lambda x: (
            _safe_int(x.get("page_index"), 0),
            _safe_int(x.get("sent_index"), 0),
            _safe_int(x.get("tok_index"), 0),
        ),
    )


def _pick_local_final_doc(final_docs: List[Dict[str, Any]], doc_id: Optional[str], filename: Optional[str]) -> Dict[str, Any]:
    for row in final_docs:
        if not isinstance(row, dict):
            continue
        row_doc_id = str(row.get("doc_id") or "").strip()
        row_filename = str(row.get("filename") or "").strip()
        if doc_id and row_doc_id and row_doc_id == doc_id:
            return row
        if filename and row_filename and (row_filename == filename or _same_filename(row_filename, filename)):
            return row
    return {}


def _pick_local_classification(results: List[Dict[str, Any]], doc_id: Optional[str], filename: Optional[str]) -> Dict[str, Any]:
    for row in results:
        if not isinstance(row, dict):
            continue
        row_doc_id = str(row.get("doc_id") or "").strip()
        row_filename = str(row.get("filename") or "").strip()
        if doc_id and row_doc_id and row_doc_id == doc_id:
            return row
        if filename and row_filename and (row_filename == filename or _same_filename(row_filename, filename)):
            return row
    return {}


def _build_local_payload_for_doc(
    ctx: Dict[str, Any],
    tok_doc: Dict[str, Any],
    final_doc: Dict[str, Any],
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    doc_id = str(tok_doc.get("doc_id") or final_doc.get("doc_id") or "").strip() or ns()
    filename = str(tok_doc.get("filename") or final_doc.get("filename") or "").strip()
    paths = _safe_list(tok_doc.get("paths"))
    if not paths and final_doc.get("source_path"):
        paths = [str(final_doc.get("source_path"))]

    pages = _normalize_pages_from_doc(tok_doc) if isinstance(tok_doc, dict) else []
    if not pages:
        pages = _pages_from_final_doc(final_doc)

    text_raw = _text_from_tok_pages(pages)
    if not text_raw.strip():
        text_raw = str(final_doc.get("text") or "")

    file_size = _resolve_file_size(
        ctx,
        paths,
        filename,
        tok_doc.get("size") or final_doc.get("size"),
    )

    doc_type = str(classification.get("doc_type") or ns())
    detected_langs = _safe_list(tok_doc.get("detected_languages")) or extract_detected_languages(ctx)
    extraction_rows = _filter_and_dedupe_extractions(ctx.get("EXTRACTIONS"), doc_id, filename)

    nlp_sentences = _filter_rows_for_doc(ctx.get("NLP_SENTENCES"), doc_id, filename)
    nlp_entities = _filter_rows_for_doc(ctx.get("NLP_ENTITIES"), doc_id, filename)
    nlp_matches = _filter_rows_for_doc(ctx.get("NLP_MATCHES"), doc_id, filename)
    nlp_tokens = _collect_local_tokens_for_doc(ctx, doc_id, filename)

    payload: Dict[str, Any] = {
        "document_id": doc_id,
        "file": {
            "path": paths,
            "name": filename,
            "size": file_size,
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
                "title": ctx.get("TITLE") or filename or None,
                "keywords": ctx.get("SEARCH_KEYWORDS") or [],
            },
        },
        "document_structure": {
            "pages_meta": extract_pages_meta(ctx),
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
            "method": final_doc.get("extraction"),
            "native": ctx.get("PREPROCESS_RESULT", {}).get("NATIVE_FILES") if isinstance(ctx.get("PREPROCESS_RESULT"), dict) else None,
            "tesseract": ctx.get("PREPROCESS_RESULT", {}).get("IMAGE_ONLY_FILES") if isinstance(ctx.get("PREPROCESS_RESULT"), dict) else None,
            "regex_extractions": extraction_rows,
            "business": ctx.get("BUSINESS") or {},
            "relations": ctx.get("RELATIONS") or [],
            "quality_checks": ctx.get("QUALITY_CHECKS") or [],
        },
        "nlp": {
            "language": ctx.get("NLP_LANGUAGE"),
            "sentences": nlp_sentences,
            "entities": nlp_entities,
            "matches": nlp_matches,
            "tokens": nlp_tokens,
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


def build_payloads_from_context(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    tok_docs = _dedupe_docs(_safe_list(ctx.get("TOK_DOCS") or ctx.get("selected")))
    final_docs = [d for d in _safe_list(ctx.get("FINAL_DOCS")) if isinstance(d, dict)]
    results = [r for r in _safe_list(ctx.get("RESULTS")) if isinstance(r, dict)]

    payloads: List[Dict[str, Any]] = []

    if tok_docs:
        for tok_doc in tok_docs:
            doc_id = str(tok_doc.get("doc_id") or "").strip() or None
            filename = str(tok_doc.get("filename") or "").strip() or None
            final_doc = _pick_local_final_doc(final_docs, doc_id, filename)
            classification = _pick_local_classification(results, doc_id, filename)
            payloads.append(_build_local_payload_for_doc(ctx, tok_doc, final_doc, classification))
        return payloads

    if final_docs:
        for row in final_docs:
            doc_stub = {
                "doc_id": row.get("doc_id"),
                "filename": row.get("filename"),
                "paths": [row.get("source_path")] if row.get("source_path") else [],
                "size": row.get("size"),
                "pages": _pages_from_final_doc(row),
                "detected_languages": [],
            }
            doc_id = str(row.get("doc_id") or "").strip() or None
            filename = str(row.get("filename") or "").strip() or None
            classification = _pick_local_classification(results, doc_id, filename)
            payloads.append(_build_local_payload_for_doc(ctx, doc_stub, row, classification))
    return payloads


def build_payload_from_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    classification = extract_classification(ctx)
    doc_type = extract_doc_type(classification)
    text_raw = extract_text_raw(ctx)
    pages_meta = extract_pages_meta(ctx)
    pages = extract_pages(ctx)
    detected_langs = extract_detected_languages(ctx)
    tok_docs = _dedupe_docs(_safe_list(ctx.get("TOK_DOCS") or ctx.get("selected")))
    first_doc = first(tok_docs) if tok_docs else {}
    first_doc = first_doc if isinstance(first_doc, dict) else {}
    file_paths = _safe_list(ctx.get("INPUT_FILE")) or _safe_list(first_doc.get("paths"))
    file_name = (
        _basename(file_paths[0])
        if file_paths
        else str(first_doc.get("filename") or "").strip() or None
    )
    file_size = _resolve_file_size(
        ctx,
        file_paths,
        file_name,
        first_doc.get("size"),
    )
    doc_id_hint = str(first_doc.get("doc_id") or "").strip() or None
    extractions = _filter_and_dedupe_extractions(
        extract_extractions(ctx),
        doc_id_hint,
        file_name,
    )
    nlp_tokens = _safe_list(ctx.get("NLP_TOKENS"))
    if not nlp_tokens:
        # Fallback compat runs anciens: fusionne NLP_POS + NLP_LEMMA.
        pos_rows = _safe_list(ctx.get("NLP_POS"))
        lemma_rows = _safe_list(ctx.get("NLP_LEMMA"))
        merged: Dict[Tuple[Any, Any, Any, Any, Any, Any], Dict[str, Any]] = {}
        for row in pos_rows:
            if not isinstance(row, dict):
                continue
            key = (
                row.get("filename"),
                row.get("page_index"),
                row.get("sent_index"),
                row.get("tok_index"),
                row.get("token"),
                row.get("lang"),
            )
            merged[key] = {
                "filename": row.get("filename"),
                "page_index": row.get("page_index"),
                "sent_index": row.get("sent_index"),
                "tok_index": row.get("tok_index"),
                "token": row.get("token"),
                "pos": row.get("pos"),
                "lemma": None,
                "ner": row.get("ner"),
                "lang": row.get("lang"),
            }
        for row in lemma_rows:
            if not isinstance(row, dict):
                continue
            key = (
                row.get("filename"),
                row.get("page_index"),
                row.get("sent_index"),
                row.get("tok_index"),
                row.get("token"),
                row.get("lang"),
            )
            cur = merged.get(key)
            if cur is None:
                merged[key] = {
                    "filename": row.get("filename"),
                    "page_index": row.get("page_index"),
                    "sent_index": row.get("sent_index"),
                    "tok_index": row.get("tok_index"),
                    "token": row.get("token"),
                    "pos": None,
                    "lemma": row.get("lemma"),
                    "ner": row.get("ner"),
                    "lang": row.get("lang"),
                }
            else:
                cur["lemma"] = row.get("lemma")
                if cur.get("ner") is None:
                    cur["ner"] = row.get("ner")
        nlp_tokens = sorted(
            merged.values(),
            key=lambda x: (
                _safe_int(x.get("page_index"), 0),
                _safe_int(x.get("sent_index"), 0),
                _safe_int(x.get("tok_index"), 0),
            ),
        )

    payload: Dict[str, Any] = {
        "document_id": ctx.get("DOC_ID") or ns(),
        "file": {
            "path": file_paths,
            "name": file_name,
            "size": file_size,
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
            "tokens": nlp_tokens,
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
    store: Any,
) -> Dict[str, Any]:
    doc_id = str(src.get("doc_id") or src.get("_id") or ns())
    filename = str(src.get("filename") or doc_id)
    paths = _safe_list(src.get("paths"))

    from_ctx_cls = _pick_from_map(cls_map, doc_id, filename)
    from_ctx_ext = _pick_from_map(ext_map, doc_id, filename)
    classification = src.get("classification") or from_ctx_cls or {}
    extraction = (
        _safe_load_json(src.get("rule_extraction_payload"))
        or src.get("rule_extraction")
        or from_ctx_ext
        or {}
    )

    doc_type = str(
        (classification.get("doc_type") if isinstance(classification, dict) else None)
        or src.get("doc_type")
        or ns()
    )

    text_raw = str(src.get("full_text") or "") or _es_text_from_pages(src)
    pages = _es_pages(src)
    words = _safe_list(src.get("words"))
    detected_langs = _safe_list(src.get("detected_languages"))
    file_size = _resolve_file_size(ctx, paths, filename, src.get("size"))

    payload: Dict[str, Any] = {
        "document_id": doc_id,
        "file": {
            "path": paths,
            "name": filename,
            "size": file_size,
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
        "nlp": _nlp_from_es_and_ctx(ctx, src, store, doc_id, filename),
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
        payloads = [
            build_payload_from_es_source(ctx, src, cls_map, ext_map, store)
            for src in es_sources
            if isinstance(src, dict)
        ]
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
        payloads = build_payloads_from_context(ctx)
        if not payloads:
            payload = build_payload_from_context(ctx)
            payloads = [payload]

    final_payload = _final_output(ctx, payloads, source)
    OUTPUT_PATH.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx["FUSION_RESULT"] = str(OUTPUT_PATH)
    ctx["FUSION_PAYLOAD"] = final_payload
    ctx["FUSION_PAYLOADS"] = final_payload.get("documents") if isinstance(final_payload, dict) else payloads
    ctx["FUSION_SOURCE"] = source
    ctx["ES_FUSION_SYNCED"] = es_synced

    # Impression et logging basique pour apparaître dans outputgeneralterminal.txt
    print("[Component: fusion-resultats]")
    print(f"[fusion-result] source={source} | docs={len(payloads)} | es_synced={es_synced}")
    print(f"[fusion-result] JSON fusionne ecrit dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
