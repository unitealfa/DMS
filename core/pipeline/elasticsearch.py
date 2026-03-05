from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF]+", flags=re.UNICODE)
LOCAL_ES_HOSTS = {"localhost", "127.0.0.1", "::1"}
AUTO_START_DEFAULT_WAIT_SECONDS = 45
AUTO_START_DEFAULT_LAUNCH_TIMEOUT = 20
ES_NLP_DEFAULT_LEVEL = "summary"
ES_NLP_DEFAULT_MAX_FULL_TOKENS = 200000
ES_NLP_BULK_BATCH = 2000
ES_NLP_SUMMARY_TOP_K = 30
ES_NLP_SUMMARY_MAX_ENTITY_SAMPLES = 60
ES_NLP_SUMMARY_MAX_SENTENCE_SAMPLES = 40
_AUTO_START_ATTEMPTED: set[str] = set()


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


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except Exception:
        pass
    return default


def _is_local_es_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in LOCAL_ES_HOSTS


def _normalize_command(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple)):
        cmd = [os.path.expandvars(str(part).strip()) for part in raw if str(part).strip()]
        return cmd
    if isinstance(raw, str):
        try:
            parsed = shlex.split(raw.strip(), posix=(os.name != "nt"))
            return [os.path.expandvars(part) for part in parsed if part.strip()]
        except Exception:
            return []
    return []


def _format_command(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _resolve_auto_start_commands(context: Dict[str, Any]) -> List[List[str]]:
    commands: List[List[str]] = []

    def _append_many(raw: Any) -> None:
        if isinstance(raw, list):
            for item in raw:
                cmd = _normalize_command(item)
                if cmd:
                    commands.append(cmd)
            return
        if isinstance(raw, str):
            txt = raw.strip()
            if not txt:
                return
            if txt.startswith("["):
                try:
                    loaded = json.loads(txt)
                except Exception:
                    loaded = None
                if isinstance(loaded, list):
                    for item in loaded:
                        cmd = _normalize_command(item)
                        if cmd:
                            commands.append(cmd)
                    return
            # Commandes en chaine dans .env:
            # ES_START_COMMANDS=cmd1 || cmd2 || cmd3
            for piece in txt.split("||"):
                cmd = _normalize_command(piece)
                if cmd:
                    commands.append(cmd)
            return

    def _append_one(raw: Any) -> None:
        cmd = _normalize_command(raw)
        if cmd:
            commands.append(cmd)

    raw_commands = context.get("ES_START_COMMANDS")
    _append_many(raw_commands)

    if not commands:
        if os.name == "nt":
            _append_one(context.get("ES_START_COMMAND_WINDOWS"))
            if not commands:
                _append_one(os.environ.get("ES_START_COMMAND_WINDOWS"))
        else:
            _append_one(context.get("ES_START_COMMAND_POSIX"))
            if not commands:
                _append_one(os.environ.get("ES_START_COMMAND_POSIX"))

    if not commands:
        _append_one(context.get("ES_START_COMMAND"))
        if not commands:
            _append_one(os.environ.get("ES_START_COMMAND"))

    if commands:
        return commands

    if os.name == "nt":
        return [
            ["powershell", "-NoProfile", "-Command", "Start-Service elasticsearch"],
            ["sc", "start", "elasticsearch"],
            ["docker", "start", "elasticsearch"],
            ["docker", "start", "es01"],
            ["docker", "compose", "up", "-d", "elasticsearch"],
            ["docker-compose", "up", "-d", "elasticsearch"],
        ]

    # Linux/macOS: pas d'auto-start par defaut (demarrage manuel).
    return []


def _run_auto_start_command(
    cmd: List[str],
    timeout_seconds: int,
    stdin_text: Optional[str] = None,
) -> tuple[bool, str]:
    if not cmd:
        return False, "commande vide"

    executable = cmd[0]
    if shutil.which(executable) is None:
        return False, f"binaire introuvable: {executable}"

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input=stdin_text,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout apres {timeout_seconds}s"
    except Exception as exc:
        return False, str(exc)

    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        details = details[:240] if details else "aucun detail"
        return False, f"code={proc.returncode} ({details})"

    return True, "ok"


def _wait_for_es_ping(store: "ElasticsearchStore", wait_seconds: int) -> bool:
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if store.ping():
            return True
        time.sleep(1.0)
    return store.ping()


def _try_auto_start_elasticsearch(store: "ElasticsearchStore", context: Dict[str, Any]) -> bool:
    if context.get("ES_AUTO_START") is False:
        return False

    if not _is_local_es_url(store.base_url):
        logging.warning(
            "Elasticsearch indisponible (%s): auto-start ignore car l'URL n'est pas locale.",
            store.base_url,
        )
        return False

    attempt_key = store.base_url
    if attempt_key in _AUTO_START_ATTEMPTED:
        return False
    _AUTO_START_ATTEMPTED.add(attempt_key)

    commands = _resolve_auto_start_commands(context)
    if not commands:
        return False

    launch_timeout = _safe_positive_int(
        context.get("ES_AUTO_START_LAUNCH_TIMEOUT"),
        AUTO_START_DEFAULT_LAUNCH_TIMEOUT,
    )
    wait_seconds = _safe_positive_int(
        context.get("ES_AUTO_START_WAIT_SECONDS"),
        AUTO_START_DEFAULT_WAIT_SECONDS,
    )

    logging.warning(
        "Elasticsearch indisponible (%s). Tentative de demarrage automatique...",
        store.base_url,
    )

    start_password = str(
        context.get("ES_START_PASSWORD") or os.environ.get("ES_START_PASSWORD") or ""
    )

    for cmd in commands:
        cmd_text = _format_command(cmd)
        stdin_text = None
        if start_password and any(part.lower() == "sudo" for part in cmd) and "-S" in cmd:
            stdin_text = f"{start_password}\n"

        ok, detail = _run_auto_start_command(
            cmd,
            timeout_seconds=launch_timeout,
            stdin_text=stdin_text,
        )
        if not ok:
            logging.info("[es-auto-start] Echec: %s | %s", cmd_text, detail)
            continue

        logging.info("[es-auto-start] Commande executee: %s", cmd_text)
        if _wait_for_es_ping(store, wait_seconds):
            context["ES_AUTO_STARTED"] = True
            context["ES_AUTO_START_CMD"] = cmd_text
            logging.info("[es-auto-start] Elasticsearch actif sur %s.", store.base_url)
            return True

        logging.info(
            "[es-auto-start] Commande ok mais ping KO apres %ss: %s",
            wait_seconds,
            cmd_text,
        )

    context["ES_AUTO_STARTED"] = False
    return False


def _same_es_target(context: Dict[str, Any], base_url: str, index: str) -> bool:
    ctx_url = str(context.get("ES_URL") or "http://localhost:9200").rstrip("/")
    ctx_index = str(context.get("ES_INDEX") or "dms_documents").strip()
    return ctx_url == base_url.rstrip("/") and ctx_index == index.strip()


def _split_words(text: str) -> List[str]:
    words = [m.group(0).lower() for m in WORD_RE.finditer(text or "")]
    return _unique_keep_order(words)


def _json_serialize(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps({"serialization_error": True, "repr": repr(value)}, ensure_ascii=False)


def _normalize_nlp_level(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    if not val:
        return ES_NLP_DEFAULT_LEVEL
    if val in {"0", "false", "none", "off", "disabled"}:
        return "off"
    if val in {"full", "all", "token", "tokens"}:
        return "full"
    return "summary"


def _normalize_nlp_index(raw: Any, base_index: str) -> str:
    idx = str(raw or "").strip()
    if idx:
        return idx
    base = str(base_index or "").strip() or "dms_documents"
    return f"{base}_nlp_tokens"


def _to_clean_str(value: Any) -> str:
    txt = str(value or "").strip()
    return txt


def _top_counter_items(counter: Counter, limit: int = ES_NLP_SUMMARY_TOP_K) -> List[Dict[str, Any]]:
    return [{"tag": tag, "count": int(count)} for tag, count in counter.most_common(limit)]


def _extract_entities(entities: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if isinstance(entities, dict):
        flat = entities.get("flat")
        if isinstance(flat, list):
            for row in flat:
                if not isinstance(row, dict):
                    continue
                etype = _to_clean_str(row.get("type"))
                etext = _to_clean_str(row.get("text"))
                if etype and etext:
                    out.append({"type": etype, "text": etext})
            return out

        for etype, vals in entities.items():
            etype_txt = _to_clean_str(etype)
            if not etype_txt or not isinstance(vals, list):
                continue
            for v in vals:
                etext = _to_clean_str(v)
                if etext:
                    out.append({"type": etype_txt, "text": etext})
        return out

    if isinstance(entities, list):
        for row in entities:
            if isinstance(row, dict):
                etype = _to_clean_str(row.get("type"))
                etext = _to_clean_str(row.get("text"))
                if etype and etext:
                    out.append({"type": etype, "text": etext})
    return out


def _entities_sample_flat(entities: List[Dict[str, str]], limit: int = ES_NLP_SUMMARY_MAX_ENTITY_SAMPLES) -> List[str]:
    out: List[str] = []
    for row in entities[:limit]:
        etype = _to_clean_str(row.get("type"))
        etext = _to_clean_str(row.get("text"))
        if not etype or not etext:
            continue
        # Prefixe type|texte pour eviter l'auto-detection "date" Elasticsearch.
        out.append(f"{etype}|{etext}")
    return out


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


def _file_size_from_paths(paths: Any) -> Optional[int]:
    if not isinstance(paths, list):
        return None
    uniq: List[str] = []
    seen = set()
    for p in paths:
        sp = _to_clean_str(p)
        if not sp or sp in seen:
            continue
        seen.add(sp)
        uniq.append(sp)
    if not uniq:
        return None
    total = 0
    found = False
    for p in uniq:
        try:
            total += int(os.path.getsize(p))
            found = True
        except Exception:
            continue
    if not found:
        return None
    return total


def _extract_doc_size(doc: Dict[str, Any]) -> Optional[int]:
    raw = doc.get("size")
    if raw is not None and raw != "":
        try:
            parsed = int(raw)
            if parsed >= 0:
                return parsed
        except Exception:
            pass
    return _file_size_from_paths(doc.get("paths"))


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
        "size": _extract_doc_size(doc),
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
    def __init__(
        self,
        base_url: str,
        index: str,
        timeout: float = 2.0,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = (base_url or "http://localhost:9200").rstrip("/")
        self.index = (index or "dms_documents").strip()
        self.timeout = timeout
        self.username = _to_clean_str(username) or None
        self.password = str(password or "")
        self.api_key = _to_clean_str(api_key) or None

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Any] = None,
        allow_404: bool = False,
        content_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
        elif self.username:
            token = base64.b64encode(
                f"{self.username}:{self.password}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        if payload is not None:
            if isinstance(payload, (bytes, bytearray)):
                data = bytes(payload)
            elif isinstance(payload, str):
                data = payload.encode("utf-8")
            else:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = content_type or "application/json"

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
                    "size": {"type": "long"},
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
                    "nlp_updated_at": {"type": "date"},
                    "nlp": {
                        "properties": {
                            "level": {"type": "keyword"},
                            "updated_at": {"type": "date"},
                            "sentences_count": {"type": "integer"},
                            "tokens_count": {"type": "integer"},
                            "entities_count": {"type": "integer"},
                            "languages": {"type": "keyword"},
                            "top_pos": {
                                "type": "nested",
                                "properties": {
                                    "tag": {"type": "keyword"},
                                    "count": {"type": "integer"},
                                },
                            },
                            "top_ner": {
                                "type": "nested",
                                "properties": {
                                    "tag": {"type": "keyword"},
                                    "count": {"type": "integer"},
                                },
                            },
                            "entities_sample": {
                                "type": "nested",
                                "properties": {
                                    "type": {"type": "keyword"},
                                    "text": {"type": "text"},
                                },
                            },
                            "entities_sample_flat": {"type": "keyword"},
                            "sentences_sample": {
                                "type": "nested",
                                "properties": {
                                    "page_index": {"type": "integer"},
                                    "sent_index": {"type": "integer"},
                                    "lang": {"type": "keyword"},
                                    "text": {"type": "text"},
                                },
                            },
                        }
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

    def ensure_custom_index(self, index: str, payload: Dict[str, Any]) -> None:
        target = str(index or "").strip()
        if not target:
            raise ValueError("Index Elasticsearch vide")
        try:
            self._request("PUT", f"/{quote(target)}", payload=payload)
        except RuntimeError as exc:
            msg = str(exc)
            if "resource_already_exists_exception" in msg:
                return
            raise

    def delete_by_query(self, index: str, query: Dict[str, Any], refresh: bool = False) -> int:
        target = str(index or "").strip()
        if not target:
            return 0
        path = f"/{quote(target)}/_delete_by_query?conflicts=proceed"
        if refresh:
            path += "&refresh=true"
        res = self._request("POST", path, payload={"query": query}) or {}
        return _safe_int(res.get("deleted"), 0)

    def bulk_ndjson(self, ndjson_payload: str, refresh: bool = False) -> Dict[str, Any]:
        path = "/_bulk"
        if refresh:
            path += "?refresh=true"
        res = self._request(
            "POST",
            path,
            payload=ndjson_payload,
            content_type="application/x-ndjson",
        ) or {}
        return res

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


def _build_nlp_doc_lookup(tok_docs: Any) -> tuple[Dict[str, str], Dict[str, List[str]]]:
    by_doc_id: Dict[str, str] = {}
    by_filename: Dict[str, List[str]] = {}
    if not isinstance(tok_docs, list):
        return by_doc_id, by_filename

    for row in tok_docs:
        if not isinstance(row, dict):
            continue
        doc_id = _to_clean_str(row.get("doc_id"))
        if not doc_id:
            continue
        by_doc_id[doc_id] = doc_id
        filename = _to_clean_str(row.get("filename"))
        if filename:
            by_filename.setdefault(filename, []).append(doc_id)

    return by_doc_id, by_filename


def _resolve_nlp_doc_id(
    row: Dict[str, Any],
    by_doc_id: Dict[str, str],
    by_filename: Dict[str, List[str]],
    store: ElasticsearchStore,
    lookup_cache: Dict[str, str],
) -> Optional[str]:
    row_doc_id = _to_clean_str(row.get("doc_id"))
    if row_doc_id and row_doc_id in by_doc_id:
        return row_doc_id

    filename = _to_clean_str(row.get("filename") or row.get("doc"))
    if filename and filename in by_filename and by_filename[filename]:
        return by_filename[filename][0]

    if not filename:
        return None

    if filename not in lookup_cache:
        found = store.find_document_id(None, filename)
        lookup_cache[filename] = _to_clean_str(found)
    cached = lookup_cache.get(filename) or ""
    return cached or None


def _ensure_nlp_tokens_index(store: ElasticsearchStore, index_name: str) -> None:
    payload = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "lang": {"type": "keyword"},
                "page_index": {"type": "integer"},
                "sent_index": {"type": "integer"},
                "tok_index": {"type": "integer"},
                "token": {"type": "keyword"},
                "token_text": {"type": "text"},
                "lemma": {"type": "keyword"},
                "pos": {"type": "keyword"},
                "ner": {"type": "keyword"},
                "sentence_text": {"type": "text"},
                "updated_at": {"type": "date"},
            }
        }
    }
    store.ensure_custom_index(index_name, payload)


def _build_nlp_token_doc_id(row: Dict[str, Any]) -> str:
    doc_id = _to_clean_str(row.get("doc_id"))
    page_index = _safe_int(row.get("page_index"), 0)
    sent_index = _safe_int(row.get("sent_index"), 0)
    tok_index = _safe_int(row.get("tok_index"), 0)
    digest = hashlib.sha1(
        f"{doc_id}|{page_index}|{sent_index}|{tok_index}".encode("utf-8")
    ).hexdigest()[:40]
    return f"nlp-{digest}"


def _bulk_index_nlp_tokens(
    store: ElasticsearchStore,
    index_name: str,
    rows: List[Dict[str, Any]],
) -> tuple[int, int]:
    indexed = 0
    errors = 0
    if not rows:
        return indexed, errors

    for i in range(0, len(rows), ES_NLP_BULK_BATCH):
        chunk = rows[i:i + ES_NLP_BULK_BATCH]
        lines: List[str] = []
        for row in chunk:
            _id = _build_nlp_token_doc_id(row)
            lines.append(_json_serialize({"index": {"_index": index_name, "_id": _id}}))
            lines.append(_json_serialize(row))
        ndjson = "\n".join(lines) + "\n"
        resp = store.bulk_ndjson(ndjson, refresh=False)
        indexed += len(chunk)
        if resp.get("errors"):
            for item in resp.get("items") or []:
                idx = item.get("index") if isinstance(item, dict) else None
                status = _safe_int(idx.get("status") if isinstance(idx, dict) else None, 200)
                if status >= 300:
                    errors += 1

    return indexed, errors


def sync_nlp_results(store: ElasticsearchStore, context: Dict[str, Any], tok_docs: Any) -> Dict[str, Any]:
    level = _normalize_nlp_level(context.get("ES_NLP_LEVEL") or os.environ.get("ES_NLP_LEVEL"))
    result = {
        "level": level,
        "docs_synced": 0,
        "tokens_indexed": 0,
        "token_index_errors": 0,
        "unresolved_rows": 0,
        "has_nlp_input": False,
        "tokens_index": None,
        "full_truncated": False,
    }

    if level == "off":
        return result

    analyses = context.get("NLP_ANALYSES")
    if not isinstance(analyses, list) or not analyses:
        return result

    result["has_nlp_input"] = True
    now = _iso_now()
    by_doc_id, by_filename = _build_nlp_doc_lookup(tok_docs)
    lookup_cache: Dict[str, str] = {}

    summary_by_doc: Dict[str, Dict[str, Any]] = {}
    full_rows_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    max_full_tokens = _safe_positive_int(
        context.get("ES_NLP_MAX_FULL_TOKENS"),
        ES_NLP_DEFAULT_MAX_FULL_TOKENS,
    )
    full_budget = max_full_tokens

    for row in analyses:
        if not isinstance(row, dict):
            continue
        target_id = _resolve_nlp_doc_id(row, by_doc_id, by_filename, store, lookup_cache)
        if not target_id:
            result["unresolved_rows"] += 1
            continue

        summary = summary_by_doc.setdefault(
            target_id,
            {
                "sentences_count": 0,
                "tokens_count": 0,
                "entities_count": 0,
                "languages": Counter(),
                "pos": Counter(),
                "ner": Counter(),
                "entities_sample": [],
                "sentences_sample": [],
                "_entity_seen": set(),
            },
        )
        summary["sentences_count"] += 1

        lang = _to_clean_str(row.get("lang"))
        if lang:
            summary["languages"][lang] += 1

        sent_text = _to_clean_str(row.get("text"))
        page_index = _safe_int(row.get("page_index") or row.get("page"), 1)
        sent_index = _safe_int(row.get("sent_index"), 0)
        if sent_text and len(summary["sentences_sample"]) < ES_NLP_SUMMARY_MAX_SENTENCE_SAMPLES:
            summary["sentences_sample"].append(
                {
                    "page_index": page_index,
                    "sent_index": sent_index,
                    "lang": lang or None,
                    "text": sent_text,
                }
            )

        tokens = [str(x) for x in (row.get("tokens") or [])]
        pos = [str(x) for x in (row.get("pos") or [])]
        lemmas = [str(x) for x in (row.get("lemmas") or [])]
        ner_labels = [str(x) for x in (row.get("ner_labels") or [])]

        tok_size = len(tokens)
        summary["tokens_count"] += tok_size
        for i in range(tok_size):
            p = pos[i] if i < len(pos) else "UNK"
            summary["pos"][p] += 1
            n = ner_labels[i] if i < len(ner_labels) else "O"
            if n and n != "O":
                summary["ner"][n] += 1

        for ent in _extract_entities(row.get("entities")):
            summary["entities_count"] += 1
            seen_key = (ent["type"], ent["text"])
            if (
                seen_key not in summary["_entity_seen"]
                and len(summary["entities_sample"]) < ES_NLP_SUMMARY_MAX_ENTITY_SAMPLES
            ):
                summary["_entity_seen"].add(seen_key)
                summary["entities_sample"].append(ent)

        if level == "full" and tok_size:
            if full_budget <= 0:
                result["full_truncated"] = True
            else:
                rows = full_rows_by_doc.setdefault(target_id, [])
                filename = _to_clean_str(row.get("filename") or row.get("doc"))
                for i in range(tok_size):
                    if full_budget <= 0:
                        result["full_truncated"] = True
                        break
                    token = tokens[i]
                    if not token:
                        continue
                    rows.append(
                        {
                            "doc_id": target_id,
                            "filename": filename or None,
                            "lang": lang or None,
                            "page_index": page_index,
                            "sent_index": sent_index,
                            "tok_index": i,
                            "token": token,
                            "token_text": token,
                            "lemma": lemmas[i] if i < len(lemmas) else token,
                            "pos": pos[i] if i < len(pos) else "UNK",
                            "ner": ner_labels[i] if i < len(ner_labels) else "O",
                            "sentence_text": sent_text or None,
                            "updated_at": now,
                        }
                    )
                    full_budget -= 1

    for doc_id, summary in summary_by_doc.items():
        languages_counter = summary.get("languages") or Counter()
        pos_counter = summary.get("pos") or Counter()
        ner_counter = summary.get("ner") or Counter()
        languages = [lang for lang, _ in languages_counter.most_common()]
        language_stats = {lang: int(cnt) for lang, cnt in languages_counter.items()}
        payload = {
            "nlp": {
                "level": level,
                "updated_at": now,
                "sentences_count": int(summary.get("sentences_count") or 0),
                "tokens_count": int(summary.get("tokens_count") or 0),
                "entities_count": int(summary.get("entities_count") or 0),
                "languages": languages,
                "language_stats": language_stats,
                "top_pos": _top_counter_items(pos_counter, ES_NLP_SUMMARY_TOP_K),
                "top_ner": _top_counter_items(ner_counter, ES_NLP_SUMMARY_TOP_K),
                "entities_sample_flat": _entities_sample_flat(summary.get("entities_sample") or []),
                "sentences_sample": summary.get("sentences_sample") or [],
            },
            "nlp_updated_at": now,
        }
        if languages:
            payload["detected_languages"] = languages
        store.update_fields(doc_id, payload)
        result["docs_synced"] += 1

    if level == "full":
        tokens_index = _normalize_nlp_index(context.get("ES_NLP_INDEX"), store.index)
        result["tokens_index"] = tokens_index
        _ensure_nlp_tokens_index(store, tokens_index)

        all_rows: List[Dict[str, Any]] = []
        for doc_id, rows in full_rows_by_doc.items():
            if rows:
                store.delete_by_query(tokens_index, {"term": {"doc_id": doc_id}})
                all_rows.extend(rows)

        indexed, errors = _bulk_index_nlp_tokens(store, tokens_index, all_rows)
        result["tokens_indexed"] = indexed
        result["token_index_errors"] = errors

    return result


def maybe_build_store(context: Dict[str, Any]) -> Optional[ElasticsearchStore]:
    if not context.get("USE_ELASTICSEARCH"):
        return None
    base_url = str(context.get("ES_URL") or "http://localhost:9200")
    index = str(context.get("ES_INDEX") or "dms_documents")
    username = _to_clean_str(context.get("ES_USER") or os.environ.get("ES_USER"))
    api_key = _to_clean_str(context.get("ES_API_KEY") or os.environ.get("ES_API_KEY"))
    if context.get("ES_PASSWORD") is None:
        password = str(os.environ.get("ES_PASSWORD") or "")
    else:
        password = str(context.get("ES_PASSWORD") or "")

    # Evite de re-tenter/re-logguer sur les appels suivants (classification/extraction/fusion)
    # quand la meme cible ES est deja marquee indisponible dans ce run.
    if context.get("ES_AVAILABLE") is False and _same_es_target(context, base_url, index):
        return None

    store = ElasticsearchStore(
        base_url=base_url,
        index=index,
        username=username or None,
        password=password,
        api_key=api_key or None,
    )
    if not store.ping():
        _try_auto_start_elasticsearch(store, context)
    if not store.ping():
        context["ES_AVAILABLE"] = False
        logging.warning("Elasticsearch indisponible (%s). Fallback sur flux local.", base_url)
        return None
    context["ES_AVAILABLE"] = True
    return store
