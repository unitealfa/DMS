from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF]+", flags=re.UNICODE)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _split_words(text: str) -> List[str]:
    words = [m.group(0).lower() for m in WORD_RE.finditer(text or "")]
    return _unique_keep_order(words)


def _json_serialize(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps({"serialization_error": True, "repr": repr(value)}, ensure_ascii=False)


def build_es_doc_id(doc: Dict[str, Any], position: int) -> str:
    existing = doc.get("doc_id")
    if existing:
        return str(existing)
    filename = str(doc.get("filename") or f"doc-{position}")
    first_path = ""
    paths = doc.get("paths")
    if isinstance(paths, list) and paths:
        first_path = str(paths[0])
    digest = hashlib.sha1(f"{filename}|{first_path}|{position}".encode("utf-8")).hexdigest()[:24]
    return f"doc-{digest}"


def _page_text_from_page(pg: Dict[str, Any]) -> str:
    txt = (
        pg.get("page_text")
        or pg.get("ocr_text")
        or pg.get("text")
        or ""
    )
    if txt:
        return str(txt)

    sent_items = pg.get("sentences_layout") or pg.get("sentences") or pg.get("chunks") or []
    parts: List[str] = []
    for s in sent_items:
        if isinstance(s, dict):
            t = s.get("text") or ""
        else:
            t = str(s)
        if t:
            parts.append(str(t))
    return "\n".join(parts)


def flatten_tok_doc_for_index(doc: Dict[str, Any]) -> Dict[str, Any]:
    pages = doc.get("pages") or []

    pages_out: List[Dict[str, Any]] = []
    passages: List[Dict[str, Any]] = []
    full_parts: List[str] = []

    for i, pg in enumerate(pages, start=1):
        if not isinstance(pg, dict):
            continue
        page_index = _safe_int(pg.get("page_index") or pg.get("page"), i)
        page_text = _page_text_from_page(pg)
        if page_text:
            full_parts.append(page_text)

        pages_out.append(
            {
                "page_index": page_index,
                "text": page_text,
                "lang": pg.get("lang"),
                "source_path": pg.get("source_path") or pg.get("path") or "",
            }
        )

        sent_items = pg.get("sentences_layout") or pg.get("sentences") or []
        if isinstance(sent_items, list) and sent_items:
            for sent in sent_items:
                if isinstance(sent, dict):
                    stxt = str(sent.get("text") or "")
                    if not stxt.strip():
                        continue
                    passages.append(
                        {
                            "page_index": page_index,
                            "text": stxt,
                            "start": _safe_int(sent.get("start"), 0),
                            "end": _safe_int(sent.get("end"), 0),
                            "layout_kind": sent.get("layout_kind"),
                        }
                    )
                else:
                    stxt = str(sent)
                    if stxt.strip():
                        passages.append(
                            {
                                "page_index": page_index,
                                "text": stxt,
                                "start": 0,
                                "end": 0,
                                "layout_kind": "plain",
                            }
                        )
        elif page_text.strip():
            passages.append(
                {
                    "page_index": page_index,
                    "text": page_text,
                    "start": 0,
                    "end": len(page_text),
                    "layout_kind": "page",
                }
            )

    full_text = "\n\n".join(part for part in full_parts if part).strip()
    words = _split_words(full_text)[:12000]
    detected_languages = _unique_keep_order(
        [str(p.get("lang")) for p in pages_out if p.get("lang")]
    )

    return {
        "doc_id": str(doc.get("doc_id") or ""),
        "filename": doc.get("filename"),
        "content": doc.get("content"),
        "extraction": doc.get("extraction"),
        "paths": doc.get("paths") or [],
        "page_count_total": doc.get("page_count_total") or len(pages_out),
        "pages": pages_out,
        "passages": passages,
        "words": words,
        "full_text": full_text,
        "detected_languages": detected_languages,
        "updated_at": _iso_now(),
    }


class ElasticsearchStore:
    def __init__(self, base_url: str, index: str, timeout: float = 2.0):
        self.base_url = (base_url or "http://localhost:9200").rstrip("/")
        self.index = (index or "dms_documents").strip()
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        allow_404: bool = False,
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = Request(url=url, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            if allow_404 and exc.code == 404:
                return None
            raise RuntimeError(f"Elasticsearch HTTP {exc.code} on {path}: {err_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Elasticsearch unreachable ({self.base_url}): {exc}") from exc

        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}

    def ping(self) -> bool:
        try:
            self._request("GET", "/")
            return True
        except Exception:
            return False

    def ensure_index(self) -> None:
        payload = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "content": {"type": "keyword"},
                    "extraction": {"type": "keyword"},
                    "full_text": {"type": "text"},
                    "words": {"type": "keyword"},
                    "passages": {
                        "type": "nested",
                        "properties": {
                            "page_index": {"type": "integer"},
                            "text": {"type": "text"},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                            "layout_kind": {"type": "keyword"},
                        },
                    },
                    "pages": {
                        "type": "nested",
                        "properties": {
                            "page_index": {"type": "integer"},
                            "text": {"type": "text"},
                            "lang": {"type": "keyword"},
                            "source_path": {"type": "keyword"},
                        },
                    },
                }
            }
        }
        try:
            self._request("PUT", f"/{quote(self.index)}", payload=payload)
        except RuntimeError as exc:
            msg = str(exc)
            if "resource_already_exists_exception" in msg:
                return
            raise

    def upsert_document(self, doc_id: str, document: Dict[str, Any], refresh: bool = False) -> None:
        path = f"/{quote(self.index)}/_update/{quote(doc_id)}"
        if refresh:
            path += "?refresh=true"
        payload = {"doc": document, "doc_as_upsert": True}
        self._request("POST", path, payload=payload)

    def update_fields(self, doc_id: str, fields: Dict[str, Any], refresh: bool = False) -> None:
        self.upsert_document(doc_id, fields, refresh=refresh)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        res = self._request("GET", f"/{quote(self.index)}/_doc/{quote(doc_id)}", allow_404=True)
        if not res or not res.get("found"):
            return None
        src = dict(res.get("_source") or {})
        src["_id"] = res.get("_id")
        return src

    def mget(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        payload = {"ids": ids}
        res = self._request("POST", f"/{quote(self.index)}/_mget", payload=payload) or {}
        out: List[Dict[str, Any]] = []
        for doc in res.get("docs") or []:
            if not doc.get("found"):
                continue
            src = dict(doc.get("_source") or {})
            src["_id"] = doc.get("_id")
            out.append(src)
        return out

    def search(self, query: Dict[str, Any], size: int = 50) -> List[Dict[str, Any]]:
        payload = {"size": size, "query": query}
        res = self._request("POST", f"/{quote(self.index)}/_search", payload=payload) or {}
        hits = (res.get("hits") or {}).get("hits") or []
        out: List[Dict[str, Any]] = []
        for h in hits:
            src = dict(h.get("_source") or {})
            src["_id"] = h.get("_id")
            out.append(src)
        return out

    def find_document_id(self, doc_id: Optional[str], filename: Optional[str]) -> Optional[str]:
        if doc_id:
            doc = self.get_document(str(doc_id))
            if doc:
                return str(doc_id)
            hits = self.search({"term": {"doc_id": str(doc_id)}}, size=1)
            if hits:
                return str(hits[0].get("_id"))

        if filename:
            hits = self.search({"term": {"filename": str(filename)}}, size=1)
            if hits:
                return str(hits[0].get("_id"))
            hits = self.search({"match_phrase": {"filename": str(filename)}}, size=1)
            if hits:
                return str(hits[0].get("_id"))
        return None


def index_tok_docs(store: ElasticsearchStore, tok_docs: Any) -> List[str]:
    if not isinstance(tok_docs, list):
        return []

    doc_ids: List[str] = []
    for i, doc in enumerate(tok_docs):
        if not isinstance(doc, dict):
            continue
        es_id = build_es_doc_id(doc, i)
        payload = flatten_tok_doc_for_index(doc)
        payload["doc_id"] = payload.get("doc_id") or es_id
        doc["doc_id"] = payload["doc_id"]
        store.upsert_document(es_id, payload)
        doc_ids.append(es_id)
    return _unique_keep_order(doc_ids)


def _group_passages_by_page(passages: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    grouped: Dict[int, List[str]] = {}
    for p in passages:
        if not isinstance(p, dict):
            continue
        text = str(p.get("text") or "")
        if not text.strip():
            continue
        page_index = _safe_int(p.get("page_index"), 1)
        grouped.setdefault(page_index, []).append(text)
    return grouped


def to_classification_docs(es_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src in es_docs:
        if not isinstance(src, dict):
            continue
        doc_id = src.get("doc_id") or src.get("_id")
        filename = src.get("filename") or str(doc_id or "document")

        pages_out: List[Dict[str, Any]] = []
        pages = src.get("pages") or []
        if isinstance(pages, list) and pages:
            for i, pg in enumerate(pages, start=1):
                if not isinstance(pg, dict):
                    continue
                text = str(pg.get("text") or "")
                pages_out.append(
                    {
                        "page_index": _safe_int(pg.get("page_index"), i),
                        "ocr_text": text,
                    }
                )

        if not pages_out:
            grouped = _group_passages_by_page(src.get("passages") or [])
            if grouped:
                for page_index in sorted(grouped.keys()):
                    pages_out.append(
                        {
                            "page_index": page_index,
                            "ocr_text": "\n".join(grouped[page_index]),
                        }
                    )

        if not pages_out:
            fallback = str(src.get("full_text") or "")
            if not fallback.strip():
                fallback = " ".join(src.get("words") or [])
            pages_out = [{"page_index": 1, "ocr_text": fallback}]

        out.append({"doc_id": doc_id, "filename": filename, "pages": pages_out})
    return out


def to_extraction_docs(es_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src in es_docs:
        if not isinstance(src, dict):
            continue
        doc_id = src.get("doc_id") or src.get("_id")
        filename = src.get("filename") or str(doc_id or "document")

        pages_out: List[Dict[str, Any]] = []
        pages = src.get("pages") or []
        if isinstance(pages, list) and pages:
            for i, pg in enumerate(pages, start=1):
                if not isinstance(pg, dict):
                    continue
                text = str(pg.get("text") or "")
                pages_out.append(
                    {
                        "page_index": _safe_int(pg.get("page_index"), i),
                        "page_text": text,
                    }
                )

        if not pages_out:
            grouped = _group_passages_by_page(src.get("passages") or [])
            if grouped:
                for page_index in sorted(grouped.keys()):
                    pages_out.append(
                        {
                            "page_index": page_index,
                            "page_text": "\n".join(grouped[page_index]),
                        }
                    )

        if not pages_out:
            fallback = str(src.get("full_text") or "")
            if not fallback.strip():
                fallback = " ".join(src.get("words") or [])
            pages_out = [{"page_index": 1, "page_text": fallback}]

        out.append({"doc_id": doc_id, "filename": filename, "pages": pages_out})
    return out


def fetch_sources_for_ids(store: ElasticsearchStore, doc_ids: List[str]) -> List[Dict[str, Any]]:
    doc_ids = _unique_keep_order([str(x) for x in doc_ids if x])
    if doc_ids:
        return store.mget(doc_ids)
    return store.search({"match_all": {}}, size=100)


def update_classification_results(store: ElasticsearchStore, results: Any) -> int:
    if not isinstance(results, list):
        return 0
    updated = 0
    now = _iso_now()
    for row in results:
        if not isinstance(row, dict):
            continue
        target_id = store.find_document_id(
            str(row.get("doc_id")) if row.get("doc_id") else None,
            str(row.get("filename")) if row.get("filename") else None,
        )
        if not target_id:
            continue
        store.update_fields(
            target_id,
            {
                "classification": row,
                "doc_type": row.get("doc_type"),
                "classification_status": row.get("status"),
                "classification_updated_at": now,
            },
        )
        updated += 1
    return updated


def update_extraction_results(store: ElasticsearchStore, extractions: Any) -> int:
    if not isinstance(extractions, list):
        return 0
    updated = 0
    now = _iso_now()
    for row in extractions:
        if not isinstance(row, dict):
            continue
        target_id = store.find_document_id(
            str(row.get("doc_id")) if row.get("doc_id") else None,
            str(row.get("filename")) if row.get("filename") else None,
        )
        if not target_id:
            continue
        fields = row.get("fields") if isinstance(row.get("fields"), dict) else {}
        matched_keys = [
            name
            for name, cfg in fields.items()
            if isinstance(cfg, dict) and isinstance(cfg.get("matches"), list) and cfg.get("matches")
        ]
        extraction_summary = {
            "doc_id": row.get("doc_id"),
            "filename": row.get("filename"),
            "doc_type": row.get("doc_type"),
            "classification_status": row.get("classification_status"),
            "fields_count": len(fields),
            "fields_with_matches": matched_keys[:200],
        }
        store.update_fields(
            target_id,
            {
                "rule_extraction": extraction_summary,
                "rule_extraction_payload": _json_serialize(row),
                "rules_fields_count": len(fields),
                "rules_fields_matched": matched_keys[:200],
                "rules_fields_payload": _json_serialize(fields),
                "ruleset_payload": _json_serialize(row.get("ruleset") or {}),
                "rules_doc_type": row.get("doc_type"),
                "rules_classification_status": row.get("classification_status"),
                "extraction_updated_at": now,
            },
        )
        updated += 1
    return updated


def maybe_build_store(context: Dict[str, Any]) -> Optional[ElasticsearchStore]:
    if not context.get("USE_ELASTICSEARCH"):
        return None
    base_url = str(context.get("ES_URL") or "http://localhost:9200")
    index = str(context.get("ES_INDEX") or "dms_documents")
    store = ElasticsearchStore(base_url=base_url, index=index)
    if not store.ping():
        logging.warning("Elasticsearch indisponible (%s). Fallback sur flux local.", base_url)
        return None
    return store
