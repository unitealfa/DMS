from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOCAL_PG_HOSTS = {"localhost", "127.0.0.1", "::1"}
AUTO_START_DEFAULT_WAIT_SECONDS = 45
AUTO_START_DEFAULT_LAUNCH_TIMEOUT = 20
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_CONNECTION_CONFIG_CANDIDATES = [
    "component/postgres/postgres_connection.py",
    "config/postgres_connection.py",
]
DEFAULT_SCHEMA_CONFIG_CANDIDATES = [
    "component/postgres/postgres_schema.py",
    "config/postgres_schema.py",
]
DEFAULT_MAPPING_VERSION = "dms-postgres-sync-v1"
_BOOTSTRAP_ATTEMPTED: set[str] = set()
_PAYLOAD_NODE_STOP_PATHS = {
    "$.document_structure.pages",
    "$.document_structure.tables",
    "$.extraction.regex_extractions",
    "$.extraction.quality_checks",
    "$.extraction.table_extraction.tables",
    "$.extraction.table_extraction.line_items",
    "$.quality_checks",
    "$.nlp.sentences",
    "$.nlp.entities",
    "$.nlp.tokens",
    "$.nlp.matches",
    "$.ml50.chunks_embeddings",
    "$.ml50.word_embeddings",
    "$.ml100.chunks_embeddings",
    "$.ml100.word_embeddings",
    "$.cross_document_analysis.links",
}


@dataclass
class PostgresConnectionConfig:
    enabled: bool
    strict_bootstrap: bool
    host: str
    port: int
    user: str
    password: str
    admin_database: str
    psql_bin: str
    pg_isready_bin: str
    connect_timeout: int
    auto_start: bool
    auto_start_wait_seconds: int
    auto_start_launch_timeout: int
    start_password: str
    start_commands: List[List[str]]
    sync_enabled: bool
    sync_strict: bool
    sync_write_fusion_audit: bool
    sync_upsert_runs: bool
    sync_upsert_documents: bool
    sync_upsert_links: bool
    schema_config_path: Path
    config_path: Path


@dataclass
class PostgresSchemaConfig:
    database_name: str
    extensions: List[str]
    tables: List[Dict[str, str]]
    indexes: List[Dict[str, str]]
    post_sql: List[str]
    config_path: Path


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_pipeline_profile(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"pipeline50ml", "50ml"}:
        return "pipeline50ml"
    if raw in {"pipeline100ml", "100ml"}:
        return "pipeline100ml"
    return "pipeline0ml"


def _active_ml_block_names(profile: Any, doc: Optional[Dict[str, Any]] = None) -> List[str]:
    normalized = _normalize_pipeline_profile(profile)
    preferred: List[str]
    if normalized == "pipeline50ml":
        preferred = ["ml50"]
    elif normalized == "pipeline100ml":
        preferred = ["ml100"]
    else:
        preferred = []

    if doc is None:
        return preferred

    out: List[str] = []
    for block_name in preferred:
        block = _safe_dict(doc.get(block_name))
        if block:
            out.append(block_name)
    return out


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _pick_first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def _json_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, (list, dict)):
        return value if value else None
    return value


def _normalize_command(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [str(part).strip() for part in raw if str(part).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            return [part for part in shlex.split(text, posix=(os.name != "nt")) if part.strip()]
        except Exception:
            return []
    return []


def _format_command(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _load_python_module(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger la configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_path(repo_root: Path, raw: Any, default: str) -> Path:
    text = str(raw or default).strip()
    path = Path(text)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _resolve_first_existing_path(repo_root: Path, raw: Any, candidates: List[str]) -> Path:
    if raw:
        return _resolve_path(repo_root, raw, candidates[0])
    for candidate in candidates:
        path = _resolve_path(repo_root, candidate, candidate)
        if path.exists():
            return path
    return _resolve_path(repo_root, candidates[0], candidates[0])


def _normalize_sql_entries(entries: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(entries, list):
        return out
    for idx, item in enumerate(entries, start=1):
        if isinstance(item, dict):
            name = str(item.get("name") or f"entry_{idx}").strip()
            sql = str(item.get("sql") or "").strip()
            if name and sql:
                out.append({"name": name, "sql": sql})
            continue
        sql = str(item or "").strip()
        if sql:
            out.append({"name": f"entry_{idx}", "sql": sql})
    return out


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_documents(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        documents = payload.get("documents")
        if isinstance(documents, list):
            return [doc for doc in documents if isinstance(doc, dict)]
        if payload.get("document_id") or payload.get("file"):
            return [payload]
    return []


def _json_sql_literal(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    tag_base = "dms_json"
    counter = 0
    marker = f"${tag_base}$"
    while marker in text:
        counter += 1
        marker = f"${tag_base}_{counter}$"
    return f"{marker}{text}{marker}"


def _quote_sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quote_sql_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_sql_table_ident(value: str) -> str:
    return ".".join(_quote_sql_ident(part) for part in str(value).split(".") if part)


def _sql_nullable_text(value: Any) -> str:
    if value is None:
        return "NULL"
    text = str(value).strip()
    if not text:
        return "NULL"
    return _quote_sql_literal(text)


def _sql_nullable_int(value: Any) -> str:
    if value is None:
        return "NULL"
    try:
        return str(int(value))
    except Exception:
        return "NULL"


def _sql_nullable_float(value: Any) -> str:
    if value is None:
        return "NULL"
    try:
        number = float(value)
    except Exception:
        return "NULL"
    if not math.isfinite(number):
        return "NULL"
    return repr(number)


def _sql_nullable_timestamp(value: Any) -> str:
    if value is None:
        return "NULL"
    text = str(value).strip()
    if not text:
        return "NULL"
    return _quote_sql_literal(text)


def _coerce_timestamp_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = text
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        datetime.fromisoformat(candidate)
        return text
    except Exception:
        return None


def _sql_value(value: Any, *, json_mode: bool = False) -> str:
    if json_mode:
        if value is None:
            return "NULL"
        return f"{_json_sql_literal(value)}::jsonb"
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NULL"
        return repr(value)
    return _quote_sql_literal(str(value))


def _stable_hash_id(prefix: str, *parts: Any) -> str:
    seed = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return _sha256_bytes(text.encode("utf-8", errors="ignore"))


def _sha256_json(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    except Exception:
        return None
    return _sha256_bytes(raw)


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        return None
    return h.hexdigest()


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text or text in {"-", "--", "—", "None", "null"}:
        return None
    text = text.replace("\xa0", " ").replace(" ", "")
    if text.count(",") and text.count("."):
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(",") and not text.count("."):
        text = text.replace(",", ".")
    text = re.sub(r"[^0-9+\-.]", "", text)
    if not text or text in {"+", "-", "."}:
        return None
    try:
        number = float(text)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _coerce_int(value: Any) -> Optional[int]:
    number = _coerce_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except Exception:
        return None


def _extract_filename(doc: Dict[str, Any]) -> str:
    file_row = _safe_dict(doc.get("file"))
    return str(file_row.get("name") or doc.get("filename") or "").strip()


def _extract_document_id(doc: Dict[str, Any], index: int) -> str:
    document_id = str(doc.get("document_id") or "").strip()
    if document_id:
        return document_id
    classification = _safe_dict(doc.get("classification"))
    doc_id = str(classification.get("doc_id") or "").strip()
    if doc_id:
        return doc_id
    filename = _extract_filename(doc)
    return _stable_hash_id("doc", filename, index)


def _extract_doc_type(doc: Dict[str, Any]) -> Optional[str]:
    classification = _safe_dict(doc.get("classification"))
    content = _safe_dict(doc.get("content"))
    value = classification.get("doc_type") or content.get("document_kind") or content.get("doc_type")
    text = str(value or "").strip()
    return text or None


def _extract_classification_status(doc: Dict[str, Any]) -> Optional[str]:
    classification = _safe_dict(doc.get("classification"))
    text = str(classification.get("status") or "").strip()
    return text or None


def _extract_content_mode(doc: Dict[str, Any]) -> Optional[str]:
    content = _safe_dict(doc.get("content"))
    text = str(content.get("content_type") or content.get("mode") or content.get("source") or "").strip()
    return text or None


def _extract_file_size(doc: Dict[str, Any]) -> Optional[int]:
    file_row = _safe_dict(doc.get("file"))
    value = file_row.get("size")
    try:
        return int(value)
    except Exception:
        return None


def _extract_page_count(doc: Dict[str, Any]) -> Optional[int]:
    file_row = _safe_dict(doc.get("file"))
    for key in ("page_count", "page_count_total"):
        try:
            value = int(file_row.get(key))
            if value >= 0:
                return value
        except Exception:
            continue
    components = _safe_dict(doc.get("components"))
    output_txt = _safe_dict(components.get("output_txt"))
    try:
        value = int(output_txt.get("pages_count"))
        if value >= 0:
            return value
    except Exception:
        pass
    structure = _safe_dict(doc.get("document_structure"))
    pages = _safe_list(structure.get("pages"))
    if pages:
        return len(pages)
    return None


def _extract_links(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    analysis = _safe_dict(payload.get("cross_document_analysis"))
    return [row for row in _safe_list(analysis.get("links")) if isinstance(row, dict)]


def _extract_file_paths(doc: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    file_row = _safe_dict(doc.get("file"))
    raw_paths = file_row.get("paths")
    if isinstance(raw_paths, list):
        for item in raw_paths:
            text = str(item or "").strip()
            if text:
                out.append(text)
    elif isinstance(raw_paths, str) and raw_paths.strip():
        out.append(raw_paths.strip())
    pretraitement = _safe_dict(_safe_dict(doc.get("components")).get("pretraitement_de_docs"))
    matched = str(pretraitement.get("matched_path") or "").strip()
    if matched:
        out.append(matched)
    seen = set()
    dedup: List[str] = []
    for item in out:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def _extract_primary_path(doc: Dict[str, Any]) -> Optional[Path]:
    for candidate in _extract_file_paths(doc):
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    for candidate in _extract_file_paths(doc):
        path = Path(candidate).expanduser()
        if path.is_absolute():
            return path
    return None


def _extract_file_ext(doc: Dict[str, Any]) -> Optional[str]:
    filename = _extract_filename(doc)
    suffix = Path(filename).suffix.strip().lower()
    return suffix or None


def _extract_file_mime(doc: Dict[str, Any]) -> Optional[str]:
    components = _safe_dict(doc.get("components"))
    pre = _safe_dict(components.get("pretraitement_de_docs"))
    value = pre.get("detected_mime")
    text = str(value or "").strip()
    return text or None


def _flatten_languages(*values: Any) -> List[str]:
    langs: List[str] = []
    seen = set()

    def add_text(raw: Any, weight: int = 1) -> None:
        text = str(raw or "").strip().lower()
        if not text:
            return
        for sep in ("|", ";", "/"):
            text = text.replace(sep, ",")
        parts = [part.strip() for part in text.split(",")] if "," in text else [text]
        for part in parts:
            if not part or part in {"unknown", "und", "none", "null", "non_specified", "n/a"}:
                continue
            if part not in seen:
                seen.add(part)
                langs.append(part)

    def add(raw: Any) -> None:
        if raw is None:
            return
        if isinstance(raw, str):
            add_text(raw)
            return
        if isinstance(raw, dict):
            for key in ("language", "lang", "code"):
                if raw.get(key):
                    add(raw.get(key))
                    return
            for key, value in raw.items():
                if isinstance(value, (int, float)) and str(key).strip():
                    add_text(str(key), int(value) if value else 1)
            return
        if isinstance(raw, list):
            for item in raw:
                add(item)

    for value in values:
        add(value)
    return langs


def _extract_languages(doc: Dict[str, Any]) -> List[str]:
    content = _safe_dict(doc.get("content"))
    nlp = _safe_dict(doc.get("nlp"))
    components = _safe_dict(doc.get("components"))
    token_layout = _safe_dict(components.get("tokenisation_layout"))
    grammar = _safe_dict(components.get("attribution_grammaticale"))
    summary = _safe_dict(_safe_dict(nlp.get("summary")))
    structure = _safe_dict(doc.get("document_structure"))
    counter: Counter = Counter()
    order: Dict[str, int] = {}

    def add(raw: Any, weight: int = 1) -> None:
        if raw is None:
            return
        if isinstance(raw, str):
            for lang in _flatten_languages(raw):
                if lang not in order:
                    order[lang] = len(order)
                counter[lang] += max(1, int(weight or 1))
            return
        if isinstance(raw, list):
            for item in raw:
                add(item, weight=weight)
            return
        if isinstance(raw, dict):
            for key in ("language", "lang", "code"):
                if raw.get(key):
                    add(raw.get(key), weight=weight)
                    return
            numeric_found = False
            for key, value in raw.items():
                if isinstance(value, (int, float)) and str(key).strip():
                    numeric_found = True
                    try:
                        item_weight = max(1, int(value))
                    except Exception:
                        item_weight = 1
                    add(str(key), weight=item_weight)
            if numeric_found:
                return

    add(content.get("detected_languages"))
    add(token_layout.get("detected_languages"))

    for page in _safe_list(structure.get("pages")):
        if not isinstance(page, dict):
            continue
        add(page.get("lang"))
        add(page.get("detected_languages"))
        for item in _safe_list(page.get("sentences_layout")):
            if not isinstance(item, dict):
                continue
            add(item.get("lang"))

    for rows in (
        _safe_list(nlp.get("sentences")),
        _safe_list(nlp.get("entities")),
        _safe_list(nlp.get("matches")),
        _safe_list(nlp.get("tokens")),
    ):
        for row in rows:
            if not isinstance(row, dict):
                continue
            add(row.get("lang") or row.get("language"))

    add(summary.get("language_stats"))
    add(summary.get("languages"))

    if not counter:
        add(grammar.get("languages"))
        add(grammar.get("language"))
        add(nlp.get("languages"))
        add(nlp.get("language"))

    langs = sorted(counter.keys(), key=lambda lang: (-counter[lang], order.get(lang, 10**9), lang))
    return langs


def _extract_language_primary(doc: Dict[str, Any]) -> Optional[str]:
    langs = _extract_languages(doc)
    return langs[0] if langs else None


def _extract_visual_flags(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = _safe_dict(doc.get("content"))
    flags = content.get("visual_flags")
    return flags if isinstance(flags, dict) and flags else None


def _extract_ocr_confidence_avg(doc: Dict[str, Any]) -> Optional[float]:
    candidates = [
        _safe_dict(doc.get("ocr")).get("confidence_avg"),
        _safe_dict(doc.get("ocr")).get("avg_confidence"),
        _safe_dict(_safe_dict(doc.get("extraction")).get("tesseract")).get("confidence_avg"),
        _safe_dict(_safe_dict(doc.get("extraction")).get("tesseract")).get("avg_confidence"),
        _safe_dict(_safe_dict(doc.get("extraction")).get("image_preprocessing")).get("confidence_avg"),
    ]
    for value in candidates:
        number = _coerce_float(value)
        if number is not None:
            return number
    return None


def _extract_table_detection_score_avg(doc: Dict[str, Any]) -> Optional[float]:
    extraction = _safe_dict(doc.get("extraction"))
    table_extraction = _safe_dict(extraction.get("table_extraction"))
    scores: List[float] = []
    for item in _safe_list(table_extraction.get("line_items")):
        if not isinstance(item, dict):
            continue
        number = _coerce_float(item.get("confidence"))
        if number is not None:
            scores.append(number)
    structure = _safe_dict(doc.get("document_structure"))
    for table in _safe_list(structure.get("tables")):
        if not isinstance(table, dict):
            continue
        for row in _safe_list(table.get("rows")):
            if not isinstance(row, dict):
                continue
            number = _coerce_float(row.get("confidence"))
            if number is not None:
                scores.append(number)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 6)


def _extract_last_error_message(doc: Dict[str, Any]) -> Optional[str]:
    processing = _safe_dict(doc.get("processing"))
    warnings = _safe_list(processing.get("warnings"))
    logs = _safe_list(processing.get("logs"))
    for collection in (warnings, logs):
        for item in collection:
            text = str(item or "").strip()
            if text:
                return text
    return None


def _has_ocr(doc: Dict[str, Any]) -> Optional[bool]:
    extraction = _safe_dict(doc.get("extraction"))
    method = str(extraction.get("method") or "").lower()
    if "ocr" in method:
        return True
    ocr = _safe_dict(doc.get("ocr"))
    if ocr:
        return True
    tesseract = _safe_dict(extraction.get("tesseract"))
    if tesseract:
        return True
    return False


def _has_tables(doc: Dict[str, Any]) -> Optional[bool]:
    structure = _safe_dict(doc.get("document_structure"))
    extraction = _safe_dict(doc.get("extraction"))
    table_extraction = _safe_dict(extraction.get("table_extraction"))
    if _safe_list(structure.get("tables")):
        return True
    if int(table_extraction.get("tables_count") or 0) > 0:
        return True
    return False


def _has_visual_marks(doc: Dict[str, Any]) -> Optional[bool]:
    structure = _safe_dict(doc.get("document_structure"))
    if _safe_list(structure.get("visual_marks")):
        return True
    flags = _extract_visual_flags(doc) or {}
    if any(_optional_bool(value) for value in flags.values()):
        return True
    visual = _safe_dict(_safe_dict(doc.get("extraction")).get("visual_detection"))
    if int(visual.get("detections_count") or 0) > 0:
        return True
    return False


def _has_links(doc: Dict[str, Any]) -> Optional[bool]:
    cross_document = _safe_dict(doc.get("cross_document"))
    return int(cross_document.get("linked_documents_count") or 0) > 0


def _has_errors(doc: Dict[str, Any]) -> Optional[bool]:
    processing = _safe_dict(doc.get("processing"))
    return bool(_safe_list(processing.get("warnings")))


def _extract_quality_score(doc: Dict[str, Any]) -> Optional[float]:
    meta = _safe_dict(doc.get("meta"))
    quality = _safe_dict(doc.get("quality"))
    for candidate in [quality.get("score"), meta.get("quality_score")]:
        number = _coerce_float(candidate)
        if number is not None:
            return number
    return None


def _extract_search_keywords(doc: Dict[str, Any]) -> Optional[List[Any]]:
    text = _safe_dict(doc.get("text"))
    search = _safe_dict(text.get("search"))
    keywords = search.get("keywords")
    return keywords if isinstance(keywords, list) and keywords else None


def _extract_search_title(doc: Dict[str, Any]) -> Optional[str]:
    text = _safe_dict(doc.get("text"))
    search = _safe_dict(text.get("search"))
    title = str(search.get("title") or "").strip()
    return title or None


def _extract_search_full_text(doc: Dict[str, Any]) -> Optional[str]:
    text = _safe_dict(doc.get("text"))
    search = _safe_dict(text.get("search"))
    value = str(search.get("full_text") or "").strip()
    return value or None


def _extract_content_source(doc: Dict[str, Any]) -> Optional[str]:
    content = _safe_dict(doc.get("content"))
    value = str(content.get("source") or "").strip()
    return value or None


def _extract_source_document_key(doc: Dict[str, Any], document_id: str, file_sha256: Optional[str], content_sha256: Optional[str]) -> str:
    filename = _extract_filename(doc)
    primary_path = _extract_primary_path(doc)
    return _stable_hash_id(
        "srcdoc",
        file_sha256 or "",
        content_sha256 or "",
        str(primary_path or ""),
        filename,
        _extract_page_count(doc) or "",
        document_id,
    )


def _extract_regex_matches_count(extraction: Dict[str, Any]) -> Optional[int]:
    rows = _safe_list(extraction.get("regex_extractions"))
    return len(rows) if rows else None


def _extract_business_keys_count(extraction: Dict[str, Any]) -> Optional[int]:
    business = extraction.get("business")
    if isinstance(business, dict):
        return len(business) or None
    if isinstance(business, list):
        return len(business) or None
    return None


def _extract_relations_count(extraction: Dict[str, Any]) -> Optional[int]:
    rows = _safe_list(extraction.get("relations"))
    return len(rows) if rows else None


def _extract_bm25_hits_count(extraction: Dict[str, Any]) -> Optional[int]:
    bm25 = _safe_dict(extraction.get("bm25"))
    for key in ("matches", "hits", "results"):
        rows = _safe_list(bm25.get(key))
        if rows:
            return len(rows)
    count = _coerce_int(bm25.get("count"))
    return count


def _extract_visual_detection_count(extraction: Dict[str, Any]) -> Optional[int]:
    visual = _safe_dict(extraction.get("visual_detection"))
    count = _coerce_int(visual.get("detections_count"))
    if count is not None:
        return count
    detections = _safe_list(visual.get("detections"))
    return len(detections) if detections else None


def _build_psql_stdin_cmd(cfg: PostgresConnectionConfig, database: Optional[str]) -> List[str]:
    return [
        cfg.psql_bin,
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-h",
        cfg.host,
        "-p",
        str(cfg.port),
        "-U",
        cfg.user,
        "-d",
        str(database or cfg.admin_database),
    ]


def load_postgres_connection_config(repo_root: Path) -> PostgresConnectionConfig:
    config_path = _resolve_first_existing_path(repo_root, None, DEFAULT_CONNECTION_CONFIG_CANDIDATES)
    module = _load_python_module(config_path, "dms_postgres_connection_config")

    raw_commands = getattr(module, "POSTGRES_START_COMMANDS", [])
    commands: List[List[str]] = []
    if isinstance(raw_commands, list):
        for item in raw_commands:
            cmd = _normalize_command(item)
            if cmd:
                commands.append(cmd)

    schema_config_path = _resolve_first_existing_path(
        repo_root,
        getattr(module, "POSTGRES_SCHEMA_CONFIG", None),
        DEFAULT_SCHEMA_CONFIG_CANDIDATES,
    )

    return PostgresConnectionConfig(
        enabled=_safe_bool(getattr(module, "POSTGRES_ENABLED", True), True),
        strict_bootstrap=_safe_bool(getattr(module, "POSTGRES_STRICT_BOOTSTRAP", False), False),
        host=str(getattr(module, "POSTGRES_HOST", "127.0.0.1")).strip() or "127.0.0.1",
        port=_safe_int(getattr(module, "POSTGRES_PORT", 5432), 5432),
        user=str(getattr(module, "POSTGRES_USER", "postgres")).strip() or "postgres",
        password=str(getattr(module, "POSTGRES_PASSWORD", "") or ""),
        admin_database=str(getattr(module, "POSTGRES_ADMIN_DATABASE", "postgres")).strip() or "postgres",
        psql_bin=str(getattr(module, "POSTGRES_PSQL_BIN", "psql")).strip() or "psql",
        pg_isready_bin=str(getattr(module, "POSTGRES_PG_ISREADY_BIN", "pg_isready")).strip() or "pg_isready",
        connect_timeout=max(1, _safe_int(getattr(module, "POSTGRES_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT), DEFAULT_CONNECT_TIMEOUT)),
        auto_start=_safe_bool(getattr(module, "POSTGRES_AUTO_START", True), True),
        auto_start_wait_seconds=max(
            1,
            _safe_int(getattr(module, "POSTGRES_AUTO_START_WAIT_SECONDS", AUTO_START_DEFAULT_WAIT_SECONDS), AUTO_START_DEFAULT_WAIT_SECONDS),
        ),
        auto_start_launch_timeout=max(
            1,
            _safe_int(getattr(module, "POSTGRES_AUTO_START_LAUNCH_TIMEOUT", AUTO_START_DEFAULT_LAUNCH_TIMEOUT), AUTO_START_DEFAULT_LAUNCH_TIMEOUT),
        ),
        start_password=str(getattr(module, "POSTGRES_START_PASSWORD", "") or ""),
        start_commands=commands,
        sync_enabled=_safe_bool(getattr(module, "POSTGRES_SYNC_ENABLED", True), True),
        sync_strict=_safe_bool(getattr(module, "POSTGRES_SYNC_STRICT", False), False),
        sync_write_fusion_audit=_safe_bool(getattr(module, "POSTGRES_SYNC_WRITE_FUSION_AUDIT", True), True),
        sync_upsert_runs=_safe_bool(getattr(module, "POSTGRES_SYNC_UPSERT_RUNS", True), True),
        sync_upsert_documents=_safe_bool(getattr(module, "POSTGRES_SYNC_UPSERT_DOCUMENTS", True), True),
        sync_upsert_links=_safe_bool(getattr(module, "POSTGRES_SYNC_UPSERT_LINKS", True), True),
        schema_config_path=schema_config_path,
        config_path=config_path,
    )


def load_postgres_schema_config(schema_config_path: Path) -> PostgresSchemaConfig:
    module = _load_python_module(schema_config_path, "dms_postgres_schema_config")
    return PostgresSchemaConfig(
        database_name=str(getattr(module, "POSTGRES_DATABASE_NAME", "dms_core")).strip() or "dms_core",
        extensions=[str(item).strip() for item in getattr(module, "POSTGRES_EXTENSIONS", []) if str(item).strip()],
        tables=_normalize_sql_entries(getattr(module, "POSTGRES_TABLES", [])),
        indexes=_normalize_sql_entries(getattr(module, "POSTGRES_INDEXES", [])),
        post_sql=[str(item).strip() for item in getattr(module, "POSTGRES_POST_SQL", []) if str(item).strip()],
        config_path=schema_config_path,
    )


def _pg_env(cfg: PostgresConnectionConfig, database: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["PGHOST"] = cfg.host
    env["PGPORT"] = str(cfg.port)
    env["PGUSER"] = cfg.user
    env["PGDATABASE"] = str(database or cfg.admin_database)
    env["PGCONNECT_TIMEOUT"] = str(cfg.connect_timeout)
    if cfg.password:
        env["PGPASSWORD"] = cfg.password
    return env


def _run_command(
    cmd: List[str],
    *,
    timeout_seconds: int,
    env: Optional[Dict[str, str]] = None,
    stdin_text: Optional[str] = None,
) -> tuple[bool, str, str]:
    if not cmd:
        return False, "", "commande vide"
    if shutil.which(cmd[0]) is None:
        return False, "", f"binaire introuvable: {cmd[0]}"
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input=stdin_text,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"timeout apres {timeout_seconds}s"
    except Exception as exc:
        return False, "", str(exc)

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"code={proc.returncode}"
        return False, stdout, detail
    return True, stdout, stderr


def _build_psql_cmd(cfg: PostgresConnectionConfig, database: Optional[str], sql: str, scalar: bool = False) -> List[str]:
    cmd = [cfg.psql_bin, "-X", "-v", "ON_ERROR_STOP=1", "-h", cfg.host, "-p", str(cfg.port), "-U", cfg.user, "-d", str(database or cfg.admin_database)]
    if scalar:
        cmd.extend(["-A", "-t", "-q"])
    cmd.extend(["-c", sql])
    return cmd


def _postgres_ping(cfg: PostgresConnectionConfig) -> bool:
    if shutil.which(cfg.pg_isready_bin):
        cmd = [
            cfg.pg_isready_bin,
            "-h",
            cfg.host,
            "-p",
            str(cfg.port),
            "-U",
            cfg.user,
            "-d",
            cfg.admin_database,
        ]
        ok, _, _ = _run_command(cmd, timeout_seconds=cfg.connect_timeout, env=_pg_env(cfg, cfg.admin_database))
        return ok

    cmd = _build_psql_cmd(cfg, cfg.admin_database, "SELECT 1;", scalar=True)
    ok, stdout, _ = _run_command(cmd, timeout_seconds=cfg.connect_timeout, env=_pg_env(cfg, cfg.admin_database))
    return ok and stdout.strip() == "1"


def _wait_for_postgres(cfg: PostgresConnectionConfig, wait_seconds: int) -> bool:
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _postgres_ping(cfg):
            return True
        time.sleep(1.0)
    return _postgres_ping(cfg)


def _resolve_start_commands(cfg: PostgresConnectionConfig) -> List[List[str]]:
    if cfg.start_commands:
        return cfg.start_commands
    if os.name == "nt":
        return [
            ["powershell", "-NoProfile", "-Command", "Get-Service | Where-Object {$_.Name -like 'postgresql*'} | Start-Service"],
            ["sc", "start", "postgresql-x64-17"],
            ["sc", "start", "postgresql-x64-16"],
        ]
    return [
        ["systemctl", "start", "postgresql"],
        ["service", "postgresql", "start"],
        ["pg_ctlcluster", "16", "main", "start"],
        ["pg_ctlcluster", "15", "main", "start"],
        ["pg_ctlcluster", "14", "main", "start"],
        ["docker", "start", "postgres"],
        ["docker", "compose", "up", "-d", "postgres"],
        ["docker-compose", "up", "-d", "postgres"],
    ]


def _try_auto_start_postgres(cfg: PostgresConnectionConfig, status: Dict[str, Any], start_if_needed: bool) -> bool:
    if not start_if_needed:
        return False
    if cfg.host.lower() not in LOCAL_PG_HOSTS:
        status["auto_start_skipped"] = "host_non_local"
        return False

    attempt_key = f"{cfg.host}:{cfg.port}"
    if attempt_key in _BOOTSTRAP_ATTEMPTED:
        return False
    _BOOTSTRAP_ATTEMPTED.add(attempt_key)

    logging.warning(
        "PostgreSQL indisponible (%s:%s). Tentative de demarrage automatique...",
        cfg.host,
        cfg.port,
    )
    status["auto_start_attempted"] = True
    status.setdefault("auto_start_commands", [])
    status.setdefault("auto_start_errors", [])

    for cmd in _resolve_start_commands(cfg):
        cmd_text = _format_command(cmd)
        status["auto_start_commands"].append(cmd_text)
        stdin_text = None
        if cfg.start_password and any(part.lower() == "sudo" for part in cmd) and "-S" in cmd:
            stdin_text = f"{cfg.start_password}\n"
        ok, _, detail = _run_command(
            cmd,
            timeout_seconds=cfg.auto_start_launch_timeout,
            env=os.environ.copy(),
            stdin_text=stdin_text,
        )
        if not ok:
            logging.info("[postgres-auto-start] Echec: %s | %s", cmd_text, detail)
            status["auto_start_errors"].append({"command": cmd_text, "error": detail})
            continue
        logging.info("[postgres-auto-start] Commande executee: %s", cmd_text)
        if _wait_for_postgres(cfg, cfg.auto_start_wait_seconds):
            status["auto_started"] = True
            status["auto_start_command"] = cmd_text
            logging.info("[postgres-auto-start] PostgreSQL actif sur %s:%s.", cfg.host, cfg.port)
            return True
        logging.info(
            "[postgres-auto-start] Commande ok mais ping KO apres %ss: %s",
            cfg.auto_start_wait_seconds,
            cmd_text,
        )

    status["auto_started"] = False
    return False


def _run_scalar_sql(cfg: PostgresConnectionConfig, database: str, sql: str) -> str:
    cmd = _build_psql_cmd(cfg, database, sql, scalar=True)
    ok, stdout, detail = _run_command(cmd, timeout_seconds=max(cfg.connect_timeout, 10), env=_pg_env(cfg, database))
    if not ok:
        raise RuntimeError(detail)
    return stdout.strip()


def _run_exec_sql(cfg: PostgresConnectionConfig, database: str, sql: str) -> None:
    cmd = _build_psql_cmd(cfg, database, sql, scalar=False)
    ok, _, detail = _run_command(cmd, timeout_seconds=max(cfg.connect_timeout, 20), env=_pg_env(cfg, database))
    if not ok:
        raise RuntimeError(detail)


def _run_exec_sql_stdin(cfg: PostgresConnectionConfig, database: str, sql: str) -> None:
    cmd = _build_psql_stdin_cmd(cfg, database)
    ok, _, detail = _run_command(cmd, timeout_seconds=max(cfg.connect_timeout, 240), env=_pg_env(cfg, database), stdin_text=sql)
    if not ok:
        raise RuntimeError(detail)


def _database_exists(cfg: PostgresConnectionConfig, database_name: str) -> bool:
    sql = f"SELECT 1 FROM pg_database WHERE datname = {_quote_sql_literal(database_name)};"
    return _run_scalar_sql(cfg, cfg.admin_database, sql) == "1"


def _create_database_if_missing(cfg: PostgresConnectionConfig, schema: PostgresSchemaConfig, status: Dict[str, Any]) -> None:
    if _database_exists(cfg, schema.database_name):
        status["database_exists"] = True
        status["database_created"] = False
        return

    sql = f"CREATE DATABASE {_quote_sql_ident(schema.database_name)} WITH ENCODING 'UTF8';"
    _run_exec_sql(cfg, cfg.admin_database, sql)
    status["database_exists"] = True
    status["database_created"] = True


def _split_schema_table(table_name: str) -> Tuple[str, str]:
    text = str(table_name or "").strip()
    if "." in text:
        schema_name, short_name = text.split(".", 1)
        return schema_name.strip() or "public", short_name.strip()
    return "public", text


def _table_exists(cfg: PostgresConnectionConfig, database_name: str, table_name: str) -> bool:
    schema_name, short_name = _split_schema_table(table_name)
    sql = (
        "SELECT 1 FROM information_schema.tables "
        f"WHERE table_schema = {_quote_sql_literal(schema_name)} AND table_name = {_quote_sql_literal(short_name)};"
    )
    return _run_scalar_sql(cfg, database_name, sql) == "1"


def _split_schema_sql_phases(sql_items: List[str]) -> Tuple[List[str], List[str]]:
    pre_table_sql: List[str] = []
    post_table_sql: List[str] = []
    for raw_sql in sql_items:
        sql = str(raw_sql or "").strip()
        if not sql:
            continue
        normalized = re.sub(r"\s+", " ", sql).strip().upper()
        if normalized.startswith("ALTER TABLE ") or " CREATE OR REPLACE VIEW " in f" {normalized} " or normalized.startswith("CREATE OR REPLACE VIEW "):
            post_table_sql.append(sql)
            continue
        pre_table_sql.append(sql)
    return pre_table_sql, post_table_sql


def _apply_schema(cfg: PostgresConnectionConfig, schema: PostgresSchemaConfig, status: Dict[str, Any]) -> None:
    applied_tables: List[str] = []
    pre_table_sql, post_table_sql = _split_schema_sql_phases(schema.post_sql)

    for ext in schema.extensions:
        ext_name = str(ext).strip()
        if not ext_name:
            continue
        _run_exec_sql(cfg, schema.database_name, f"CREATE EXTENSION IF NOT EXISTS {_quote_sql_ident(ext_name)};")

    for sql in pre_table_sql:
        if sql:
            _run_exec_sql(cfg, schema.database_name, sql)

    for entry in schema.tables:
        sql = str(entry.get("sql") or "").strip()
        name = str(entry.get("name") or "").strip()
        if not sql or not name:
            continue
        _run_exec_sql(cfg, schema.database_name, sql)
        if _table_exists(cfg, schema.database_name, name):
            applied_tables.append(name)

    for sql in post_table_sql:
        if sql:
            _run_exec_sql(cfg, schema.database_name, sql)

    for entry in schema.indexes:
        sql = str(entry.get("sql") or "").strip()
        if sql:
            _run_exec_sql(cfg, schema.database_name, sql)

    status["tables_expected"] = [entry["name"] for entry in schema.tables if entry.get("name")]
    status["tables_ready"] = applied_tables
    status["tables_missing"] = [name for name in status["tables_expected"] if name not in applied_tables]
    status["schema_ready"] = len(status["tables_missing"]) == 0


def _finalize_or_raise(status: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    if strict and not status.get("ready"):
        raise RuntimeError(str(status.get("error") or "Bootstrap PostgreSQL en echec"))
    return status


def _finalize_sync_or_raise(status: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    if strict and status.get("error"):
        raise RuntimeError(str(status["error"]))
    return status


def ensure_postgres_bootstrap(
    repo_root: Path,
    *,
    start_if_needed: Optional[bool] = None,
    strict_override: Optional[bool] = None,
) -> Dict[str, Any]:
    cfg = load_postgres_connection_config(repo_root)
    schema = load_postgres_schema_config(cfg.schema_config_path)
    strict = cfg.strict_bootstrap if strict_override is None else bool(strict_override)

    status: Dict[str, Any] = {
        "enabled": cfg.enabled,
        "ready": False,
        "host": cfg.host,
        "port": cfg.port,
        "user": cfg.user,
        "admin_database": cfg.admin_database,
        "database": schema.database_name,
        "config_path": str(cfg.config_path),
        "schema_config_path": str(schema.config_path),
        "sync_enabled": cfg.sync_enabled,
        "auto_start_attempted": False,
        "auto_started": False,
        "auto_start_commands": [],
        "auto_start_errors": [],
        "database_exists": False,
        "database_created": False,
        "schema_ready": False,
        "tables_expected": [entry["name"] for entry in schema.tables if entry.get("name")],
        "tables_ready": [],
        "tables_missing": [],
        "error": None,
    }

    if not cfg.enabled:
        status["ready"] = False
        status["skipped"] = "disabled"
        return status

    if shutil.which(cfg.psql_bin) is None:
        status["error"] = f"psql introuvable: {cfg.psql_bin}"
        logging.warning("[postgres] %s", status["error"])
        return _finalize_or_raise(status, strict)

    try:
        ping_ok = _postgres_ping(cfg)
        if not ping_ok:
            _try_auto_start_postgres(cfg, status, cfg.auto_start if start_if_needed is None else bool(start_if_needed))
            ping_ok = _postgres_ping(cfg)
        if not ping_ok:
            status["error"] = f"PostgreSQL indisponible sur {cfg.host}:{cfg.port}"
            logging.warning("[postgres] %s", status["error"])
            return _finalize_or_raise(status, strict)

        _create_database_if_missing(cfg, schema, status)
        _apply_schema(cfg, schema, status)
        status["ready"] = bool(status.get("database_exists")) and bool(status.get("schema_ready"))
        logging.info(
            "[postgres] ready=%s | host=%s | port=%s | db=%s | created=%s | tables=%s/%s",
            status["ready"],
            cfg.host,
            cfg.port,
            schema.database_name,
            status["database_created"],
            len(status["tables_ready"]),
            len(status["tables_expected"]),
        )
        return _finalize_or_raise(status, strict)
    except Exception as exc:
        status["error"] = str(exc)
        logging.warning("[postgres] bootstrap en echec: %s", exc)
        return _finalize_or_raise(status, strict)


def _build_upsert_statement(
    table_name: str,
    row: Dict[str, Any],
    *,
    conflict_columns: List[str],
    json_columns: Optional[List[str]] = None,
    skip_update_columns: Optional[List[str]] = None,
) -> str:
    json_set = set(json_columns or [])
    skip_update = set(skip_update_columns or [])
    columns = list(row.keys())
    values_sql = [_sql_value(row.get(column), json_mode=(column in json_set)) for column in columns]
    update_columns = [column for column in columns if column not in conflict_columns and column not in skip_update]
    assignments = ",\n                ".join(
        f"{_quote_sql_ident(column)} = EXCLUDED.{_quote_sql_ident(column)}"
        for column in update_columns
    )
    conflict_sql = ", ".join(_quote_sql_ident(column) for column in conflict_columns)
    insert_sql = (
        f"INSERT INTO {_quote_sql_table_ident(table_name)} (\n"
        f"                {', '.join(_quote_sql_ident(column) for column in columns)}\n"
        f"            ) VALUES (\n"
        f"                {', '.join(values_sql)}\n"
        f"            )"
    )
    if assignments:
        insert_sql += f"\n            ON CONFLICT ({conflict_sql}) DO UPDATE SET\n                {assignments};"
    else:
        insert_sql += f"\n            ON CONFLICT ({conflict_sql}) DO NOTHING;"
    return insert_sql


def _append_upsert_rows(
    statements: List[str],
    table_name: str,
    rows: List[Dict[str, Any]],
    *,
    conflict_columns: List[str],
    json_columns: Optional[List[str]] = None,
    skip_update_columns: Optional[List[str]] = None,
) -> int:
    count = 0
    for row in rows:
        if not isinstance(row, dict) or not row:
            continue
        statements.append(
            _build_upsert_statement(
                table_name,
                row,
                conflict_columns=conflict_columns,
                json_columns=json_columns,
                skip_update_columns=skip_update_columns,
            )
        )
        count += 1
    return count


def _run_payload_view(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "documents"}


def _build_run_row(
    payload: Dict[str, Any],
    *,
    run_id: str,
    profile: str,
    source_value: str,
    fusion_path: Optional[Path],
    now_iso: str,
) -> Dict[str, Any]:
    profile = _normalize_pipeline_profile(profile)
    completed_at = str(payload.get("generated_at") or now_iso).strip() or now_iso
    started_at = completed_at
    pipeline = _safe_dict(payload.get("pipeline"))
    postgres_sync = _safe_dict(payload.get("postgres_sync"))
    status = "completed"
    if postgres_sync.get("error"):
        status = "partial"
    return {
        "run_id": run_id,
        "pipeline_profile": profile,
        "profile": _normalize_pipeline_profile(payload.get("profile") or profile),
        "pipeline_version": None,
        "mapping_version": DEFAULT_MAPPING_VERSION,
        "payload_schema_version": str(payload.get("schema_version") or "").strip() or None,
        "source": source_value,
        "generated_at": completed_at,
        "documents_count": len(_payload_documents(payload)),
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "fusion_path": str(fusion_path) if fusion_path else None,
        "raw_payload": _run_payload_view(payload),
        "pipeline_json": pipeline or None,
        "postgres_sync_json": postgres_sync or None,
        "null_policy_json": _json_or_none(payload.get("null_policy_json") or payload.get("null_policy")),
        "source_context_json": _json_or_none(payload.get("source_context_json") or payload.get("source_context")),
        "cross_document_analysis_json": _json_or_none(payload.get("cross_document_analysis")),
        "registries_json": _json_or_none(payload.get("registries_json") or payload.get("registries")),
        "item_templates_json": _json_or_none(payload.get("item_templates_json") or payload.get("item_templates")),
        "sql_mapping_hints_json": _json_or_none(payload.get("sql_mapping_hints_json") or payload.get("sql_mapping_hints")),
        "created_at": now_iso,
        "updated_at": now_iso,
    }


def _build_ingest_row(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    file_sha256: Optional[str],
    now_iso: str,
) -> Dict[str, Any]:
    filename = _extract_filename(doc)
    source_path = _extract_primary_path(doc)
    ingest_key = _stable_hash_id("ingest", run_id, document_id, filename)
    return {
        "ingest_key": ingest_key,
        "run_id": run_id,
        "document_id": document_id,
        "source_file_path": str(source_path) if source_path else None,
        "source_filename": filename or None,
        "file_sha256": file_sha256,
        "payload_json": doc,
        "ingest_status": "processed",
        "error_message": None,
        "received_at": now_iso,
        "processed_at": now_iso,
    }


def _build_document_row(
    doc: Dict[str, Any],
    *,
    run_id: str,
    profile: str,
    source_value: str,
    document_id: str,
    now_iso: str,
) -> Dict[str, Any]:
    profile = _normalize_pipeline_profile(profile)
    classification = _safe_dict(doc.get("classification"))
    content = _safe_dict(doc.get("content"))
    processing = _safe_dict(doc.get("processing"))
    file_sha256 = _sha256_file(_extract_primary_path(doc))
    content_sha256 = _pick_first_non_empty(
        _sha256_text(_safe_dict(doc.get("text")).get("text_raw")),
        _sha256_json(doc),
    )
    source_document_key = _extract_source_document_key(doc, document_id, file_sha256, content_sha256)
    warnings = _safe_list(processing.get("warnings"))
    logs = _safe_list(processing.get("logs"))
    return {
        "document_id": document_id,
        "run_id": run_id,
        "source_document_key": source_document_key,
        "file_sha256": file_sha256,
        "content_sha256": content_sha256,
        "document_version": _coerce_int(_safe_dict(doc.get("meta")).get("document_version")),
        "is_latest": None,
        "pipeline_profile": profile,
        "source": source_value,
        "file_name": _extract_filename(doc) or None,
        "file_path_primary": str(_extract_primary_path(doc) or "") or None,
        "file_paths_json": _json_or_none(_extract_file_paths(doc)),
        "file_size": _extract_file_size(doc),
        "file_page_count": _extract_page_count(doc),
        "file_mime": _extract_file_mime(doc),
        "file_ext": _extract_file_ext(doc),
        "file_content_mode": _extract_content_mode(doc),
        "doc_type": _extract_doc_type(doc),
        "classification_doc_id": str(classification.get("doc_id") or "").strip() or None,
        "classification_status": _extract_classification_status(doc),
        "classification_winning_score": _coerce_float(classification.get("winning_score")),
        "classification_threshold": _coerce_float(classification.get("threshold")),
        "classification_margin": _coerce_float(classification.get("margin")),
        "content_document_kind": str(content.get("document_kind") or "").strip() or None,
        "content_content_type": str(content.get("content_type") or "").strip() or None,
        "content_source": _extract_content_source(doc),
        "language_primary": _extract_language_primary(doc),
        "languages_json": _extract_languages(doc),
        "ingest_status": "processed",
        "parse_status": "completed",
        "normalization_status": "normalized" if _json_or_none(_safe_dict(doc.get("text")).get("normalization")) else None,
        "document_quality_score": _extract_quality_score(doc),
        "ocr_confidence_avg": _extract_ocr_confidence_avg(doc),
        "table_detection_score_avg": _extract_table_detection_score_avg(doc),
        "has_ocr": _has_ocr(doc),
        "has_tables": _has_tables(doc),
        "has_visual_marks": _has_visual_marks(doc),
        "has_links": _has_links(doc),
        "has_errors": _has_errors(doc),
        "last_error_message": _extract_last_error_message(doc),
        "warnings_json": warnings or None,
        "logs_json": logs or None,
        "visual_flags_json": _extract_visual_flags(doc),
        "classification_scores_json": _json_or_none(classification.get("scores")),
        "classification_keyword_matches_json": _json_or_none(classification.get("keyword_matches")),
        "classification_log_json": _json_or_none(classification.get("classification_log")),
        "classification_scores_audit_json": _json_or_none(classification.get("scores_audit")),
        "anti_confusion_targets_json": _json_or_none(classification.get("anti_confusion_targets")),
        "classification_decision_debug_json": _json_or_none(classification.get("decision_debug")),
        "created_at": now_iso,
        "updated_at": now_iso,
    }


def _build_document_text_row(doc: Dict[str, Any], *, run_id: str, document_id: str, now_iso: str) -> Dict[str, Any]:
    text = _safe_dict(doc.get("text"))
    return {
        "document_id": document_id,
        "run_id": run_id,
        "language": _extract_language_primary(doc),
        "source_title": _extract_filename(doc) or None,
        "text_title": _extract_search_title(doc),
        "text_raw": _pick_first_non_empty(text.get("text_raw"), _extract_search_full_text(doc)),
        "text_normalized": text.get("text_normalized"),
        "search_full_text": _extract_search_full_text(doc),
        "search_keywords_json": _extract_search_keywords(doc),
        "normalization_json": _json_or_none(text.get("normalization")),
        "text_json": text or None,
        "created_at": now_iso,
        "updated_at": now_iso,
    }


def _build_document_payload_row(doc: Dict[str, Any], *, run_id: str, document_id: str, now_iso: str) -> Dict[str, Any]:
    return {
        "document_id": document_id,
        "run_id": run_id,
        "payload_schema_version": None,
        "raw_document_json": doc,
        "source_payload_json": _json_or_none(doc.get("source_payload")),
        "file_json": _json_or_none(doc.get("file")),
        "classification_json": _json_or_none(doc.get("classification")),
        "content_json": _json_or_none(doc.get("content")),
        "text_json": _json_or_none(doc.get("text")),
        "document_structure_json": _json_or_none(doc.get("document_structure")),
        "extraction_json": _json_or_none(doc.get("extraction")),
        "nlp_json": _json_or_none(doc.get("nlp")),
        "ml50_json": _json_or_none(doc.get("ml50")),
        "ml100_json": _json_or_none(doc.get("ml100")),
        "components_json": _json_or_none(doc.get("components")),
        "quality_checks_json": _json_or_none(doc.get("quality_checks")),
        "cross_document_json": _json_or_none(doc.get("cross_document")),
        "processing_json": _json_or_none(doc.get("processing")),
        "meta_json": _json_or_none(doc.get("meta")),
        "ocr_json": _json_or_none(doc.get("ocr")),
        "human_review_json": _json_or_none(doc.get("human_review")),
        "created_at": now_iso,
        "updated_at": now_iso,
    }


def _build_document_extraction_summary_row(doc: Dict[str, Any], *, run_id: str, document_id: str, now_iso: str) -> Dict[str, Any]:
    extraction = _safe_dict(doc.get("extraction"))
    table_extraction = _safe_dict(extraction.get("table_extraction"))
    totals_verification = _safe_dict(extraction.get("totals_verification"))
    return {
        "document_id": document_id,
        "run_id": run_id,
        "extraction_method": str(extraction.get("method") or "").strip() or None,
        "regex_matches_count": _extract_regex_matches_count(extraction),
        "business_keys_count": _extract_business_keys_count(extraction),
        "relations_count": _extract_relations_count(extraction),
        "bm25_hits_count": _extract_bm25_hits_count(extraction),
        "tables_count": _coerce_int(table_extraction.get("tables_count")),
        "table_rows_total": _coerce_int(table_extraction.get("rows_total")),
        "totals_verification_status": str(totals_verification.get("verification_status") or "").strip() or None,
        "totals_verification_passed": _optional_bool(totals_verification.get("passed")),
        "totals_verification_complete": _optional_bool(totals_verification.get("complete")),
        "visual_detections_count": _extract_visual_detection_count(extraction),
        "native_json": _json_or_none(extraction.get("native")),
        "tesseract_json": _json_or_none(extraction.get("tesseract")),
        "regex_extractions_json": _json_or_none(extraction.get("regex_extractions")),
        "business_json": _json_or_none(extraction.get("business")),
        "relations_json": _json_or_none(extraction.get("relations")),
        "bm25_json": _json_or_none(extraction.get("bm25")),
        "table_extraction_json": _json_or_none(extraction.get("table_extraction")),
        "totals_verification_json": _json_or_none(extraction.get("totals_verification")),
        "visual_detection_json": _json_or_none(extraction.get("visual_detection")),
        "extraction_json": extraction or None,
        "created_at": now_iso,
        "updated_at": now_iso,
    }


def _text_excerpt(value: Any, limit: int = 400) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _json_path_append_key(parent_path: str, key: str) -> str:
    key_text = str(key)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key_text):
        return f"{parent_path}.{key_text}" if parent_path else key_text
    return f"{parent_path}[{json.dumps(key_text, ensure_ascii=False)}]" if parent_path else json.dumps(key_text, ensure_ascii=False)


def _json_path_append_index(parent_path: str, index: int) -> str:
    return f"{parent_path}[{index}]"


def _source_section_from_json_path(json_path: str) -> str:
    text = str(json_path or "").strip()
    if not text or text == "$":
        return "root"
    if text.startswith("$."):
        text = text[2:]
    elif text.startswith("$"):
        text = text[1:]
    if not text:
        return "root"
    if text.startswith('["'):
        end = text.find('"]')
        if end > 2:
            try:
                return str(json.loads(text[: end + 2])).strip() or "root"
            except Exception:
                pass
    text = text.lstrip(".")
    first = re.split(r"[.\[]", text, maxsplit=1)[0].strip()
    return first or "root"


def _json_node_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "scalar"


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "text"
    return "json"


def _payload_scalar_columns(value: Any) -> Dict[str, Any]:
    value_type = _json_value_type(value)
    if value_type == "null":
        return {
            "value_type": "null",
            "value_text": None,
            "value_number": None,
            "value_boolean": None,
            "value_json": None,
        }
    if value_type == "boolean":
        return {
            "value_type": "boolean",
            "value_text": str(bool(value)).lower(),
            "value_number": None,
            "value_boolean": bool(value),
            "value_json": value,
        }
    if value_type == "number":
        number = _coerce_float(value)
        return {
            "value_type": "number",
            "value_text": str(value),
            "value_number": number,
            "value_boolean": None,
            "value_json": value,
        }
    if value_type == "text":
        return {
            "value_type": "text",
            "value_text": str(value),
            "value_number": _coerce_float(value),
            "value_boolean": _optional_bool(value) if str(value).strip().lower() in {"true", "false", "1", "0", "yes", "no", "on", "off"} else None,
            "value_json": value,
        }
    return {
        "value_type": "json",
        "value_text": None,
        "value_number": None,
        "value_boolean": None,
        "value_json": value,
    }


def _walk_payload_nodes(
    value: Any,
    *,
    run_id: str,
    document_id: Optional[str],
    now_iso: str,
    table_kind: str,
    json_path: str = "$",
    parent_node_id: Optional[str] = None,
    key_name: Optional[str] = None,
    array_index: Optional[int] = None,
    source_section: Optional[str] = None,
    visited: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    node_kind = _json_node_kind(value)
    node_source_section = str(source_section or _source_section_from_json_path(json_path)).strip() or "root"
    node_id = _stable_hash_id("payloadnode", table_kind, run_id, document_id or "", json_path)
    scalar_columns = _payload_scalar_columns(value)
    rows.append(
        {
            "node_id": node_id,
            "run_id": run_id,
            "document_id": document_id,
            "parent_node_id": parent_node_id,
            "json_path": json_path,
            "path_hash": _sha256_text(json_path),
            "key_name": key_name,
            "array_index": array_index,
            "node_kind": node_kind,
            "value_type": scalar_columns["value_type"] if node_kind in {"scalar", "null"} else "json",
            "value_text": scalar_columns["value_text"] if node_kind in {"scalar", "null"} else None,
            "value_number": scalar_columns["value_number"] if node_kind == "scalar" else None,
            "value_boolean": scalar_columns["value_boolean"] if node_kind == "scalar" else None,
            "value_json": value,
            "source_section": node_source_section,
            "payload_json": value,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
    )

    if value is None or node_kind == "scalar":
        return rows

    if json_path in _PAYLOAD_NODE_STOP_PATHS:
        return rows

    if visited is None:
        visited = set()
    obj_id = id(value)
    if obj_id in visited:
        return rows
    visited.add(obj_id)

    if isinstance(value, dict):
        for child_key, child_value in value.items():
            child_key_text = str(child_key)
            child_path = _json_path_append_key(json_path, child_key_text)
            rows.extend(
                _walk_payload_nodes(
                    child_value,
                    run_id=run_id,
                    document_id=document_id,
                    now_iso=now_iso,
                    table_kind=table_kind,
                    json_path=child_path,
                    parent_node_id=node_id,
                    key_name=child_key_text,
                    array_index=None,
                    source_section=node_source_section if json_path != "$" else child_key_text,
                    visited=visited,
                )
            )
        return rows

    if isinstance(value, list):
        if value and all(not isinstance(item, (dict, list)) for item in value) and len(value) > 64:
            return rows
        for idx, child_value in enumerate(value):
            child_path = _json_path_append_index(json_path, idx)
            rows.extend(
                _walk_payload_nodes(
                    child_value,
                    run_id=run_id,
                    document_id=document_id,
                    now_iso=now_iso,
                    table_kind=table_kind,
                    json_path=child_path,
                    parent_node_id=node_id,
                    key_name=None,
                    array_index=idx,
                    source_section=node_source_section,
                    visited=visited,
                )
            )
    return rows


def _build_run_payload_node_rows(payload: Dict[str, Any], *, run_id: str, now_iso: str) -> List[Dict[str, Any]]:
    return _walk_payload_nodes(
        _run_payload_view(payload),
        run_id=run_id,
        document_id=None,
        now_iso=now_iso,
        table_kind="run",
    )


def _build_document_payload_node_rows(doc: Dict[str, Any], *, run_id: str, document_id: str, now_iso: str) -> List[Dict[str, Any]]:
    return _walk_payload_nodes(
        doc,
        run_id=run_id,
        document_id=document_id,
        now_iso=now_iso,
        table_kind="document",
    )


def _iter_leaf_values(value: Any, prefix: str = "", depth: int = 0, max_depth: int = 8) -> List[Tuple[str, Any]]:
    if depth > max_depth:
        return [(prefix or "value", value)]
    if isinstance(value, dict):
        if not value:
            return [(prefix or "value", value)]
        out: List[Tuple[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_iter_leaf_values(child, child_prefix, depth + 1, max_depth=max_depth))
        return out
    if isinstance(value, list):
        if not value:
            return [(prefix or "value", value)]
        if all(not isinstance(item, (dict, list)) for item in value):
            if len(value) > 64:
                return [(prefix or "value", value)]
            return [(_json_path_append_index(prefix or "value", idx), item) for idx, item in enumerate(value)]
        out = []
        for idx, child in enumerate(value):
            child_prefix = _json_path_append_index(prefix or "value", idx)
            out.extend(_iter_leaf_values(child, child_prefix, depth + 1, max_depth=max_depth))
        return out
    return [(prefix or "value", value)]


def _extract_item_text_excerpt(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        for key in ("text", "text_excerpt", "page_text", "snippet", "label", "title", "value", "term", "token", "name"):
            value = item.get(key)
            text = _text_excerpt(value)
            if text:
                return text
        raw_cells = item.get("raw_cells")
        if isinstance(raw_cells, list) and raw_cells:
            return _text_excerpt(" | ".join(str(x or "") for x in raw_cells))
    elif isinstance(item, list):
        return _text_excerpt(" | ".join(str(x or "") for x in item))
    else:
        return _text_excerpt(item)
    return None


def _normalize_identifier_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = re.sub(r"[^A-Z0-9]+", "", text.upper())
    return normalized or text.upper()


def _is_identifier_like_field(field_name: str) -> bool:
    text = str(field_name or "").strip().lower()
    if not text:
        return False
    markers = (
        "id",
        "identifier",
        "reference",
        "ref",
        "number",
        "numero",
        "num",
        "invoice",
        "facture",
        "contract",
        "contrat",
        "commande",
        "order",
        "bc",
        "doc",
        "dossier",
    )
    return any(marker in text for marker in markers)


def _parse_compact_keyword_match(raw: Any) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {"keyword": None, "count": None, "score": None, "bucket": None}
    match = re.match(r"^(.*?)\(x(\d+)(?:,([+-]?\d+))?(?:,([^)]*))?\)$", text)
    if not match:
        return {"keyword": text, "count": None, "score": None, "bucket": None}
    keyword, count_text, score_text, bucket_text = match.groups()
    return {
        "keyword": keyword.strip() or text,
        "count": _coerce_int(count_text),
        "score": _coerce_float(score_text),
        "bucket": str(bucket_text or "").strip() or None,
    }


def _make_registry_row(
    *,
    entity_kind: str,
    natural_key_parts: List[Any],
    stable_id: Optional[str],
    run_id: str,
    now_iso: str,
    payload_json: Any = None,
) -> Optional[Dict[str, Any]]:
    stable = str(stable_id or "").strip()
    if not stable:
        return None
    natural_key = "|".join(str(part or "") for part in natural_key_parts)
    natural_key_hash = _sha256_text(natural_key)
    if not natural_key_hash:
        return None
    return {
        "registry_id": _stable_hash_id("registry", entity_kind, natural_key_hash),
        "entity_kind": entity_kind,
        "natural_key_hash": natural_key_hash,
        "stable_id": stable,
        "first_seen_run_id": run_id,
        "last_seen_run_id": run_id,
        "payload_json": payload_json,
        "created_at": now_iso,
        "updated_at": now_iso,
    }


def _build_document_identifier_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    classification = _safe_dict(doc.get("classification"))
    text = _safe_dict(doc.get("text"))
    extraction = _safe_dict(doc.get("extraction"))
    file_sha256 = _sha256_file(_extract_primary_path(doc))
    content_sha256 = _pick_first_non_empty(_sha256_text(text.get("text_raw")), _sha256_json(doc))
    source_document_key = _extract_source_document_key(doc, document_id, file_sha256, content_sha256)

    def add_identifier(
        identifier_type: str,
        value: Any,
        *,
        identifier_scope: Optional[str] = None,
        page_index: Any = None,
        sent_index: Any = None,
        source_component: Optional[str] = None,
        source_path: Optional[str] = None,
        confidence_score: Any = None,
        payload_json: Any = None,
    ) -> None:
        value_text = str(value or "").strip()
        if not value_text:
            return
        normalized = _normalize_identifier_value(value_text) or value_text
        seen_key = (str(identifier_type or "").strip().lower(), normalized)
        if seen_key in seen:
            return
        seen.add(seen_key)
        rows.append(
            {
                "identifier_id": _stable_hash_id("identifier", run_id, document_id, identifier_type, normalized),
                "document_id": document_id,
                "run_id": run_id,
                "identifier_type": str(identifier_type or "").strip() or None,
                "identifier_scope": str(identifier_scope or "").strip() or None,
                "page_index": _coerce_int(page_index),
                "sent_index": _coerce_int(sent_index),
                "value_text": value_text,
                "value_normalized": normalized,
                "source_component": str(source_component or "").strip() or None,
                "source_path": str(source_path or "").strip() or None,
                "confidence_score": _coerce_float(confidence_score),
                "payload_json": payload_json,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    add_identifier("document_id", document_id, identifier_scope="system", source_component="postgres_sync", source_path="document_id")
    add_identifier("classification_doc_id", classification.get("doc_id"), identifier_scope="classification", source_component="classification", source_path="classification.doc_id")
    add_identifier("source_document_key", source_document_key, identifier_scope="system", source_component="postgres_sync", source_path="source_document_key")
    add_identifier("file_sha256", file_sha256, identifier_scope="file", source_component="postgres_sync", source_path="file_sha256")
    add_identifier("content_sha256", content_sha256, identifier_scope="content", source_component="postgres_sync", source_path="content_sha256")

    for regex_index, regex_row in enumerate(_safe_list(extraction.get("regex_extractions")), start=1):
        if not isinstance(regex_row, dict):
            continue
        for field_name, field_payload in _safe_dict(regex_row.get("fields")).items():
            if not _is_identifier_like_field(field_name):
                continue
            field = field_payload if isinstance(field_payload, dict) else {"value": field_payload}
            matches = [m for m in _safe_list(field.get("matches")) if isinstance(m, dict)]
            if matches:
                for match_index, match in enumerate(matches, start=1):
                    add_identifier(
                        field_name,
                        match.get("value"),
                        identifier_scope="regex",
                        page_index=match.get("page_index"),
                        source_component="extraction_regles",
                        source_path=f"extraction.regex_extractions[{regex_index - 1}].fields.{field_name}.matches[{match_index - 1}]",
                        payload_json=match,
                    )
            else:
                add_identifier(
                    field_name,
                    field.get("value"),
                    identifier_scope="regex",
                    source_component="extraction_regles",
                    source_path=f"extraction.regex_extractions[{regex_index - 1}].fields.{field_name}",
                    payload_json=field,
                )

    business = extraction.get("business")
    if isinstance(business, dict):
        for field_name, field_payload in business.items():
            if not _is_identifier_like_field(field_name):
                continue
            if isinstance(field_payload, dict):
                add_identifier(
                    field_name,
                    _pick_first_non_empty(field_payload.get("value"), field_payload.get("text"), field_payload.get("normalized_value")),
                    identifier_scope="business",
                    source_component="extraction_business",
                    source_path=f"extraction.business.{field_name}",
                    confidence_score=field_payload.get("confidence"),
                    payload_json=field_payload,
                )
            else:
                add_identifier(
                    field_name,
                    field_payload,
                    identifier_scope="business",
                    source_component="extraction_business",
                    source_path=f"extraction.business.{field_name}",
                    payload_json={"value": field_payload},
                )

    return rows


def _build_document_classification_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    classification = _safe_dict(doc.get("classification"))
    scores_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    keyword_rows: List[Dict[str, Any]] = []
    anti_hits_rows: List[Dict[str, Any]] = []
    anti_targets_rows: List[Dict[str, Any]] = []

    scores = _safe_dict(classification.get("scores"))
    winning_doc_type = str(classification.get("doc_type") or "").strip()
    ranked = sorted(scores.items(), key=lambda item: (_coerce_float(item[1]) or 0.0), reverse=True)
    for rank, (doc_type, score) in enumerate(ranked, start=1):
        scores_rows.append(
            {
                "classification_score_id": _stable_hash_id("cls-score", run_id, document_id, doc_type),
                "document_id": document_id,
                "run_id": run_id,
                "doc_type": str(doc_type),
                "score": _coerce_float(score),
                "score_rank": rank,
                "is_winner": str(doc_type).strip() == winning_doc_type if winning_doc_type else None,
                "payload_json": {"doc_type": doc_type, "score": score},
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    for doc_type, audit in _safe_dict(classification.get("scores_audit")).items():
        if not isinstance(audit, dict):
            audit_rows.append(
                {
                    "score_audit_id": _stable_hash_id("cls-audit", run_id, document_id, doc_type, "value"),
                    "document_id": document_id,
                    "run_id": run_id,
                    "doc_type": str(doc_type),
                    "metric_name": "value",
                    "metric_value_text": str(audit),
                    "metric_value_number": _coerce_float(audit),
                    "metric_value_boolean": _optional_bool(audit),
                    "payload_json": {"value": audit},
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            continue
        audit_rows.append(
            {
                "score_audit_id": _stable_hash_id("cls-audit", run_id, document_id, doc_type, "score_total"),
                "document_id": document_id,
                "run_id": run_id,
                "doc_type": str(doc_type),
                "metric_name": "score_total",
                "metric_value_text": str(audit.get("score_total")) if audit.get("score_total") is not None else None,
                "metric_value_number": _coerce_float(audit.get("score_total")),
                "metric_value_boolean": None,
                "payload_json": audit,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
        for kw_index, keyword in enumerate(_safe_list(audit.get("matched_keywords")), start=1):
            if not isinstance(keyword, dict):
                keyword = _parse_compact_keyword_match(keyword)
            audit_rows.append(
                {
                    "score_audit_id": _stable_hash_id("cls-audit", run_id, document_id, doc_type, "kw", kw_index, keyword.get("keyword") or ""),
                    "document_id": document_id,
                    "run_id": run_id,
                    "doc_type": str(doc_type),
                    "metric_name": f"matched_keyword.{str(keyword.get('bucket') or 'unknown')}",
                    "metric_value_text": str(keyword.get("keyword") or "").strip() or None,
                    "metric_value_number": _coerce_float(_pick_first_non_empty(keyword.get("score"), keyword.get("count"))),
                    "metric_value_boolean": None,
                    "payload_json": keyword,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )

    keyword_matches = _safe_dict(classification.get("keyword_matches"))
    for bucket_name, items in keyword_matches.items():
        if bucket_name == "anti_confusion_hits":
            for hit_index, item in enumerate(_safe_list(items), start=1):
                hit = item if isinstance(item, dict) else _parse_compact_keyword_match(item)
                anti_hits_rows.append(
                    {
                        "anti_confusion_hit_id": _stable_hash_id("anti-hit", run_id, document_id, hit_index, hit.get("keyword") or ""),
                        "document_id": document_id,
                        "run_id": run_id,
                        "target_doc_type": str(hit.get("target_doc_type") or hit.get("doc_type") or "").strip() or None,
                        "keyword_text": str(hit.get("keyword") or "").strip() or None,
                        "hit_count": _coerce_int(hit.get("count")),
                        "payload_json": hit,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    }
                )
            continue
        for item_index, item in enumerate(_safe_list(items), start=1):
            match = item if isinstance(item, dict) else _parse_compact_keyword_match(item)
            keyword_rows.append(
                {
                    "keyword_match_id": _stable_hash_id("cls-keyword", run_id, document_id, bucket_name, item_index, match.get("keyword") or ""),
                    "document_id": document_id,
                    "run_id": run_id,
                    "bucket_name": bucket_name,
                    "doc_type": str(match.get("doc_type") or "").strip() or None,
                    "keyword_text": str(match.get("keyword") or "").strip() or None,
                    "match_count": _coerce_int(match.get("count")),
                    "weight": _coerce_float(_pick_first_non_empty(match.get("score"), match.get("weight"))),
                    "payload_json": match,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )

    for target_index, target in enumerate(_safe_list(classification.get("anti_confusion_targets")), start=1):
        target_text = str(target or "").strip()
        if not target_text:
            continue
        anti_targets_rows.append(
            {
                "anti_confusion_target_id": _stable_hash_id("anti-target", run_id, document_id, target_text),
                "document_id": document_id,
                "run_id": run_id,
                "target_doc_type": target_text,
                "target_rank": target_index,
                "payload_json": {"target": target_text},
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    return scores_rows, audit_rows, keyword_rows, anti_hits_rows, anti_targets_rows


def _build_document_page_meta_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    structure = _safe_dict(doc.get("document_structure"))
    rows: List[Dict[str, Any]] = []
    for ordinal, meta in enumerate(_safe_list(structure.get("pages_meta")), start=1):
        raw = meta if isinstance(meta, dict) else {"value": meta}
        page_index = _coerce_int(_pick_first_non_empty(raw.get("page_index"), ordinal))
        rows.append(
            {
                "page_meta_id": _stable_hash_id("pagemeta", run_id, document_id, page_index or ordinal),
                "document_id": document_id,
                "run_id": run_id,
                "page_id": _stable_hash_id("page", run_id, document_id, page_index or ordinal),
                "page_index": page_index,
                "width": _coerce_float(_pick_first_non_empty(raw.get("width"), raw.get("page_width"))),
                "height": _coerce_float(_pick_first_non_empty(raw.get("height"), raw.get("page_height"))),
                "rotation": _coerce_float(raw.get("rotation")),
                "dpi": _coerce_float(raw.get("dpi")),
                "lang": str(raw.get("lang") or "").strip() or None,
                "source_path": str(raw.get("source_path") or "").strip() or None,
                "payload_json": raw,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_document_sentence_layout_row_sets(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
    valid_sentence_ids: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    structure = _safe_dict(doc.get("document_structure"))
    layout_rows: List[Dict[str, Any]] = []
    span_rows: List[Dict[str, Any]] = []
    header_rows: List[Dict[str, Any]] = []
    header_cell_rows: List[Dict[str, Any]] = []
    table_row_rows: List[Dict[str, Any]] = []

    for page_ordinal, page in enumerate(_safe_list(structure.get("pages")), start=1):
        if not isinstance(page, dict):
            continue
        page_index = _coerce_int(_pick_first_non_empty(page.get("page_index"), page_ordinal))
        for layout_ordinal, layout in enumerate(_safe_list(page.get("sentences_layout")), start=1):
            raw = layout if isinstance(layout, dict) else {"text": layout}
            text = str(raw.get("text") or "").strip()
            sent_index = _coerce_int(_pick_first_non_empty(raw.get("sent_index"), layout_ordinal - 1))
            sentence_id = None
            if text:
                candidate_sentence_id = _stable_hash_id("sentence", run_id, document_id, page_index or "", sent_index or "", text)
                if valid_sentence_ids is None or candidate_sentence_id in valid_sentence_ids:
                    sentence_id = candidate_sentence_id
            sentence_layout_id = _stable_hash_id("sentlayout", run_id, document_id, page_index or "", sent_index or "", layout_ordinal, text)
            layout_rows.append(
                {
                    "sentence_layout_id": sentence_layout_id,
                    "document_id": document_id,
                    "run_id": run_id,
                    "sentence_id": sentence_id,
                    "page_index": page_index,
                    "sent_index": sent_index,
                    "line": _coerce_int(raw.get("line")),
                    "col": _coerce_float(raw.get("col")),
                    "col_index": _coerce_int(raw.get("col_index")),
                    "layout_kind": str(raw.get("layout_kind") or "").strip() or None,
                    "is_sentence": _optional_bool(raw.get("is_sentence")),
                    "is_noise": _optional_bool(raw.get("is_noise")),
                    "nonspace": _coerce_int(raw.get("nonspace")),
                    "source_path": str(page.get("source_path") or "").strip() or None,
                    "payload_json": raw,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            for span_index, span in enumerate(_safe_list(raw.get("spans")), start=1):
                start_offset = None
                end_offset = None
                bbox_json = None
                if isinstance(span, dict):
                    start_offset = _coerce_int(_pick_first_non_empty(span.get("start"), span.get("char_start"), span.get("from")))
                    end_offset = _coerce_int(_pick_first_non_empty(span.get("end"), span.get("char_end"), span.get("to")))
                    bbox_json = span.get("bbox")
                    span_payload = span
                elif isinstance(span, (list, tuple)) and len(span) >= 2:
                    start_offset = _coerce_int(span[0])
                    end_offset = _coerce_int(span[1])
                    span_payload = {"start": start_offset, "end": end_offset}
                else:
                    span_payload = {"value": span}
                span_text = None
                if text and start_offset is not None and end_offset is not None and 0 <= start_offset <= end_offset <= len(text):
                    span_text = text[start_offset:end_offset]
                span_rows.append(
                    {
                        "sentence_span_id": _stable_hash_id("sentspan", run_id, document_id, sentence_layout_id, span_index, start_offset or "", end_offset or ""),
                        "document_id": document_id,
                        "run_id": run_id,
                        "sentence_id": sentence_id,
                        "sentence_layout_id": sentence_layout_id,
                        "span_index": span_index,
                        "page_index": page_index,
                        "line": _coerce_int(raw.get("line")),
                        "col": _coerce_float(raw.get("col")),
                        "col_index": _coerce_int(raw.get("col_index")),
                        "start_offset": start_offset,
                        "end_offset": end_offset,
                        "text": span_text or _extract_item_text_excerpt(span_payload),
                        "bbox_json": bbox_json,
                        "payload_json": span_payload,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    }
                )

            for header_row_index, header_row in enumerate(_safe_list(raw.get("header_rows")), start=1):
                header_payload = header_row if isinstance(header_row, dict) else {"text": header_row}
                header_row_id = _stable_hash_id("headerrow", run_id, document_id, sentence_layout_id, header_row_index, _extract_item_text_excerpt(header_payload) or "")
                header_rows.append(
                    {
                        "header_row_id": header_row_id,
                        "document_id": document_id,
                        "run_id": run_id,
                        "page_index": page_index,
                        "table_index": _coerce_int(header_payload.get("table_index")),
                        "row_index": _coerce_int(_pick_first_non_empty(header_payload.get("row_index"), header_row_index - 1)),
                        "text": _extract_item_text_excerpt(header_payload),
                        "payload_json": header_payload,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    }
                )
                for cell_index, cell in enumerate(_safe_list(header_payload.get("cells")), start=1):
                    cell_payload = cell if isinstance(cell, dict) else {"text": cell}
                    header_cell_rows.append(
                        {
                            "header_cell_id": _stable_hash_id("headercell", run_id, document_id, header_row_id, cell_index, _extract_item_text_excerpt(cell_payload) or ""),
                            "document_id": document_id,
                            "run_id": run_id,
                            "header_row_id": header_row_id,
                            "page_index": page_index,
                            "table_index": _coerce_int(cell_payload.get("table_index")),
                            "row_index": _coerce_int(_pick_first_non_empty(cell_payload.get("row_index"), header_row_index - 1)),
                            "col_index": _coerce_int(_pick_first_non_empty(cell_payload.get("col_index"), cell_index - 1)),
                            "text": _extract_item_text_excerpt(cell_payload),
                            "normalized_text": str(cell_payload.get("normalized_text") or _extract_item_text_excerpt(cell_payload) or "").strip() or None,
                            "payload_json": cell_payload,
                            "created_at": now_iso,
                            "updated_at": now_iso,
                        }
                    )

            for table_row_index, table_row in enumerate(_safe_list(raw.get("table_rows")), start=1):
                table_payload = table_row if isinstance(table_row, dict) else {"text": table_row}
                table_row_rows.append(
                    {
                        "layout_table_row_id": _stable_hash_id("layouttable", run_id, document_id, sentence_layout_id, table_row_index, _extract_item_text_excerpt(table_payload) or ""),
                        "document_id": document_id,
                        "run_id": run_id,
                        "document_table_id": None,
                        "page_index": page_index,
                        "table_index": _coerce_int(table_payload.get("table_index")),
                        "row_index": _coerce_int(_pick_first_non_empty(table_payload.get("row_index"), table_row_index - 1)),
                        "line": _coerce_int(_pick_first_non_empty(table_payload.get("line"), raw.get("line"))),
                        "col": _coerce_float(_pick_first_non_empty(table_payload.get("col"), raw.get("col"))),
                        "col_index": _coerce_int(_pick_first_non_empty(table_payload.get("col_index"), raw.get("col_index"))),
                        "layout_kind": str(table_payload.get("layout_kind") or raw.get("layout_kind") or "").strip() or None,
                        "is_sentence": _optional_bool(_pick_first_non_empty(table_payload.get("is_sentence"), raw.get("is_sentence"))),
                        "is_noise": _optional_bool(_pick_first_non_empty(table_payload.get("is_noise"), raw.get("is_noise"))),
                        "nonspace": _coerce_int(_pick_first_non_empty(table_payload.get("nonspace"), raw.get("nonspace"))),
                        "text": _extract_item_text_excerpt(table_payload),
                        "payload_json": table_payload,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    }
                )

    return layout_rows, span_rows, header_rows, header_cell_rows, table_row_rows


def _build_document_text_aux_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    text = _safe_dict(doc.get("text"))
    search = _safe_dict(text.get("search"))
    normalization_rows: List[Dict[str, Any]] = []
    keyword_rows: List[Dict[str, Any]] = []

    for item_index, item in enumerate(_safe_list(text.get("normalization")), start=1):
        payload = item if isinstance(item, dict) else {"value": item}
        normalization_rows.append(
            {
                "normalization_item_id": _stable_hash_id("textnorm", run_id, document_id, item_index, _extract_item_text_excerpt(payload) or ""),
                "document_id": document_id,
                "run_id": run_id,
                "item_index": item_index,
                "field_name": str(payload.get("field_name") or payload.get("field") or payload.get("name") or "").strip() or None,
                "lang": str(payload.get("lang") or "").strip() or None,
                "original_text": str(_pick_first_non_empty(payload.get("original_text"), payload.get("original"), payload.get("raw"), payload.get("source")) or "").strip() or None,
                "normalized_text": str(_pick_first_non_empty(payload.get("normalized_text"), payload.get("normalized"), payload.get("value")) or "").strip() or None,
                "method": str(payload.get("method") or "").strip() or None,
                "source_path": str(payload.get("source_path") or "").strip() or None,
                "payload_json": payload,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    for keyword_rank, item in enumerate(_safe_list(search.get("keywords")), start=1):
        payload = item if isinstance(item, dict) else {"keyword": item}
        keyword_rows.append(
            {
                "search_keyword_id": _stable_hash_id("searchkw", run_id, document_id, keyword_rank, payload.get("keyword") or payload.get("term") or item),
                "document_id": document_id,
                "run_id": run_id,
                "keyword_rank": keyword_rank,
                "keyword_text": str(payload.get("keyword") or payload.get("term") or item or "").strip() or None,
                "score": _coerce_float(payload.get("score")),
                "source_path": str(payload.get("source_path") or "text.search.keywords").strip() or None,
                "payload_json": payload,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    return normalization_rows, keyword_rows


def _build_document_business_field_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    extraction = _safe_dict(doc.get("extraction"))
    business = extraction.get("business")
    if not business:
        return []
    rows: List[Dict[str, Any]] = []

    if isinstance(business, dict):
        iterator = list(business.items())
    else:
        iterator = [(f"item_{index}", value) for index, value in enumerate(_safe_list(business), start=1)]

    for field_name, field_payload in iterator:
        payload = field_payload if isinstance(field_payload, dict) else {"value": field_payload}
        value = _pick_first_non_empty(
            payload.get("value"),
            payload.get("text"),
            payload.get("number"),
            payload.get("amount"),
            payload.get("date"),
            field_payload if not isinstance(field_payload, dict) else None,
        )
        scalar = _payload_scalar_columns(value)
        rows.append(
            {
                "business_field_id": _stable_hash_id("business", run_id, document_id, field_name),
                "document_id": document_id,
                "run_id": run_id,
                "field_group": str(str(field_name).split(".", 1)[0]).strip() or None,
                "field_name": str(field_name),
                "source_component": "extraction_business",
                "source_path": f"extraction.business.{field_name}",
                "value_text": scalar["value_text"],
                "value_number": scalar["value_number"],
                "value_boolean": scalar["value_boolean"],
                "value_json": field_payload if isinstance(field_payload, (dict, list)) else None,
                "confidence_score": _coerce_float(payload.get("confidence")) if isinstance(payload, dict) else None,
                "payload_json": field_payload,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_document_human_review_task_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    human_review = _safe_dict(doc.get("human_review"))
    rows: List[Dict[str, Any]] = []
    for task_index, task in enumerate(_safe_list(human_review.get("tasks")), start=1):
        payload = task if isinstance(task, dict) else {"title": task}
        rows.append(
            {
                "human_review_task_id": _stable_hash_id("reviewtask", run_id, document_id, task_index, _extract_item_text_excerpt(payload) or ""),
                "document_id": document_id,
                "run_id": run_id,
                "task_index": task_index,
                "task_type": str(payload.get("task_type") or payload.get("type") or "").strip() or None,
                "status": str(payload.get("status") or "").strip() or None,
                "title": _extract_item_text_excerpt(payload),
                "assignee": str(payload.get("assignee") or "").strip() or None,
                "due_at": _coerce_timestamp_text(payload.get("due_at")),
                "payload_json": payload,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_document_processing_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    processing = _safe_dict(doc.get("processing"))
    warning_rows: List[Dict[str, Any]] = []
    log_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    duration_rows: List[Dict[str, Any]] = []

    for warning_index, warning in enumerate(_safe_list(processing.get("warnings")), start=1):
        payload = warning if isinstance(warning, dict) else {"message": warning}
        warning_rows.append(
            {
                "processing_warning_id": _stable_hash_id("procwarn", run_id, document_id, warning_index, _extract_item_text_excerpt(payload) or ""),
                "document_id": document_id,
                "run_id": run_id,
                "warning_index": warning_index,
                "level": str(payload.get("level") or "warning").strip() or None,
                "code": str(payload.get("code") or "").strip() or None,
                "message": _extract_item_text_excerpt(payload),
                "source_component": str(payload.get("component") or payload.get("source_component") or "").strip() or None,
                "payload_json": payload,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    for log_index, log in enumerate(_safe_list(processing.get("logs")), start=1):
        payload = log if isinstance(log, dict) else {"message": log}
        log_rows.append(
            {
                "processing_log_id": _stable_hash_id("proclog", run_id, document_id, log_index, _extract_item_text_excerpt(payload) or ""),
                "document_id": document_id,
                "run_id": run_id,
                "log_index": log_index,
                "level": str(payload.get("level") or "info").strip() or None,
                "message": _extract_item_text_excerpt(payload),
                "source_component": str(payload.get("component") or payload.get("source_component") or "").strip() or None,
                "logged_at": _coerce_timestamp_text(_pick_first_non_empty(payload.get("timestamp"), payload.get("logged_at"))),
                "payload_json": payload,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    for step_index, step in enumerate(_safe_list(processing.get("pipeline")), start=1):
        payload = step if isinstance(step, dict) else {"step_name": step}
        step_rows.append(
            {
                "processing_step_id": _stable_hash_id("procstep", run_id, document_id, step_index, payload.get("step_name") or payload.get("name") or step),
                "document_id": document_id,
                "run_id": run_id,
                "step_index": step_index,
                "step_name": str(payload.get("step_name") or payload.get("name") or step or "").strip() or None,
                "source_component": str(payload.get("component") or payload.get("source_component") or payload.get("name") or step or "").strip() or None,
                "status": str(payload.get("status") or "completed").strip() or None,
                "started_at": _coerce_timestamp_text(payload.get("started_at")),
                "completed_at": _coerce_timestamp_text(payload.get("completed_at")),
                "duration_ms": _coerce_float(_pick_first_non_empty(payload.get("duration_ms"), payload.get("duration"), payload.get("ms"))),
                "payload_json": payload,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    for metric_name, metric_value in _iter_leaf_values(_safe_dict(processing.get("durations"))):
        scalar = _payload_scalar_columns(metric_value)
        duration_rows.append(
            {
                "processing_duration_id": _stable_hash_id("procduration", run_id, document_id, metric_name),
                "document_id": document_id,
                "run_id": run_id,
                "metric_name": metric_name,
                "source_component": metric_name.split(".", 1)[0] if "." in metric_name else None,
                "duration_ms": _coerce_float(_pick_first_non_empty(metric_value if isinstance(metric_value, (int, float, str)) else None, scalar["value_number"])),
                "payload_json": metric_value,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    return warning_rows, log_rows, step_rows, duration_rows


def _build_document_component_metric_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    components = _safe_dict(doc.get("components"))
    rows: List[Dict[str, Any]] = []
    for component_name, payload in components.items():
        if not isinstance(payload, dict):
            payload = {"value": payload}
        for metric_name, metric_value in _iter_leaf_values(payload):
            scalar = _payload_scalar_columns(metric_value)
            rows.append(
                {
                    "component_metric_id": _stable_hash_id("compmetric", run_id, document_id, component_name, metric_name),
                    "document_id": document_id,
                    "run_id": run_id,
                    "component_name": str(component_name),
                    "metric_name": metric_name,
                    "metric_value_text": scalar["value_text"],
                    "metric_value_number": scalar["value_number"],
                    "metric_value_boolean": scalar["value_boolean"],
                    "metric_value_json": metric_value if isinstance(metric_value, (dict, list)) else None,
                    "source_path": f"components.{component_name}.{metric_name}",
                    "payload_json": metric_value,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
    return rows


def _collect_link_term_examples(link: Dict[str, Any], term: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    examples_a: List[Dict[str, Any]] = []
    examples_b: List[Dict[str, Any]] = []
    audit = _safe_dict(link.get("audit"))
    needle = str(term or "").strip().lower()
    if not needle:
        return examples_a, examples_b
    for match in _safe_list(audit.get("matches")):
        if not isinstance(match, dict):
            continue
        shared_terms = {str(item or "").strip().lower() for item in _safe_list(match.get("shared_terms"))}
        if needle not in shared_terms:
            continue
        phrase_a = _safe_dict(match.get("phrase_a"))
        phrase_b = _safe_dict(match.get("phrase_b"))
        if phrase_a:
            examples_a.append(phrase_a)
        if phrase_b:
            examples_b.append(phrase_b)
    return examples_a[:3], examples_b[:3]


def _component_status_text(component_payload: Dict[str, Any]) -> Optional[str]:
    for key in ("status", "verification_status", "doc_type", "source", "engine", "output_path", "method"):
        text = str(component_payload.get(key) or "").strip()
        if text:
            return text
    return None


def _build_document_component_audit_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    components = _safe_dict(doc.get("components"))
    rows: List[Dict[str, Any]] = []
    for component_name, payload in components.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "component_audit_id": _stable_hash_id("compaudit", run_id, document_id, component_name),
                "document_id": document_id,
                "run_id": run_id,
                "component_name": component_name,
                "backend": str(payload.get("backend") or payload.get("engine") or payload.get("source") or "").strip() or None,
                "method": str(
                    payload.get("method")
                    or payload.get("pos_method")
                    or payload.get("nlp_level")
                    or payload.get("embedding_method")
                    or ""
                ).strip() or None,
                "model": str(payload.get("model") or "").strip() or None,
                "status_text": _component_status_text(payload),
                "payload_json": payload,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_document_page_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    structure = _safe_dict(doc.get("document_structure"))
    rows: List[Dict[str, Any]] = []
    for ordinal, page in enumerate(_safe_list(structure.get("pages")), start=1):
        if not isinstance(page, dict):
            continue
        page_index = _coerce_int(_pick_first_non_empty(page.get("page_index"), ordinal))
        page_text = _pick_first_non_empty(page.get("page_text"), page.get("text"))
        rows.append(
            {
                "page_id": _stable_hash_id("page", run_id, document_id, page_index or ordinal),
                "document_id": document_id,
                "run_id": run_id,
                "page_index": page_index,
                "lang": str(page.get("lang") or "").strip() or None,
                "chars": _coerce_int(page.get("chars")),
                "source_path": str(page.get("source_path") or "").strip() or None,
                "page_text": str(page_text) if page_text is not None else None,
                "text_excerpt": _text_excerpt(page_text),
                "raw_page_json": page,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_generic_structure_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
    block_name: str,
    table_name: str,
    id_prefix: str,
    pk_name: str,
) -> List[Dict[str, Any]]:
    structure = _safe_dict(doc.get("document_structure"))
    out: List[Dict[str, Any]] = []
    for ordinal, item in enumerate(_safe_list(structure.get(block_name)), start=1):
        raw = item if isinstance(item, dict) else {"value": item}
        page_index = _coerce_int(_pick_first_non_empty(raw.get("page_index"), raw.get("page"), None))
        out.append(
            {
                pk_name: _stable_hash_id(id_prefix, run_id, document_id, block_name, page_index or "", ordinal, _extract_item_text_excerpt(raw) or ""),
                "document_id": document_id,
                "run_id": run_id,
                "page_index": page_index,
                "item_index": ordinal,
                "text_excerpt": _extract_item_text_excerpt(raw),
                "raw_json": raw,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return out


def _build_document_visual_mark_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    structure = _safe_dict(doc.get("document_structure"))
    rows: List[Dict[str, Any]] = []
    for ordinal, mark in enumerate(_safe_list(structure.get("visual_marks")), start=1):
        if not isinstance(mark, dict):
            continue
        rows.append(
            {
                "visual_mark_id": _stable_hash_id(
                    "vmark",
                    run_id,
                    document_id,
                    mark.get("page_index") or "",
                    mark.get("type") or "",
                    ordinal,
                ),
                "document_id": document_id,
                "run_id": run_id,
                "page_index": _coerce_int(mark.get("page_index")),
                "mark_type": str(mark.get("type") or "").strip() or None,
                "kind": str(mark.get("kind") or mark.get("type") or "").strip() or None,
                "score": _coerce_float(mark.get("score")),
                "confidence": _coerce_float(_pick_first_non_empty(mark.get("confidence"), mark.get("score"))),
                "source": str(mark.get("source") or "").strip() or None,
                "engine": str(mark.get("engine") or mark.get("source") or "").strip() or None,
                "decoded_value": str(mark.get("decoded_value") or mark.get("value") or "").strip() or None,
                "page_width": _coerce_int(mark.get("page_width")),
                "page_height": _coerce_int(mark.get("page_height")),
                "bbox_px_json": _json_or_none(mark.get("bbox_px")),
                "bbox_norm_json": _json_or_none(mark.get("bbox_norm")),
                "payload_json": mark,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_document_structure_detail_row_sets(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> Dict[str, List[Dict[str, Any]]]:
    specs = [
        ("sections", "dms.document_sections", "section", "section_id"),
        ("blocks", "dms.document_blocks", "block", "block_id"),
        ("lines", "dms.document_lines", "line", "line_id"),
        ("words", "dms.document_words", "word", "word_id"),
        ("headers", "dms.document_headers", "header", "header_id"),
        ("footers", "dms.document_footers", "footer", "footer_id"),
        ("lists", "dms.document_lists", "list", "list_id"),
        ("figures", "dms.document_figures", "figure", "figure_id"),
        ("equations", "dms.document_equations", "equation", "equation_id"),
        ("key_value_pairs", "dms.document_key_value_pairs", "kv", "key_value_id"),
        ("reading_order", "dms.document_reading_order", "reading", "reading_order_id"),
        ("non_text_regions", "dms.document_non_text_regions", "nontext", "non_text_region_id"),
    ]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for block_name, table_name, prefix, pk_name in specs:
        out[table_name] = _build_generic_structure_rows(
            doc,
            run_id=run_id,
            document_id=document_id,
            now_iso=now_iso,
            block_name=block_name,
            table_name=table_name,
            id_prefix=prefix,
            pk_name=pk_name,
        )
    return out


def _build_document_topic_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    profile: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    normalized_profile = _normalize_pipeline_profile(profile)
    for block_name in _active_ml_block_names(normalized_profile, doc):
        block = _safe_dict(doc.get(block_name))
        if not block:
            continue
        primary_terms = {str(term).strip() for term in _safe_list(block.get("document_primary_topics")) if str(term).strip()}
        for rank, topic in enumerate(_safe_list(block.get("document_topics")), start=1):
            if isinstance(topic, dict):
                term = str(topic.get("term") or "").strip()
                score = _coerce_float(topic.get("score"))
                payload = topic
            else:
                term = str(topic or "").strip()
                score = None
                payload = {"term": topic}
            if not term:
                continue
            rows.append(
                {
                    "topic_id": _stable_hash_id("topic", run_id, document_id, block_name, "document", rank, term),
                    "document_id": document_id,
                    "run_id": run_id,
                    "pipeline_profile": normalized_profile,
                    "topic_scope": "document",
                    "topic_source": normalized_profile,
                    "page_index": None,
                    "sent_index": None,
                    "topic_rank": rank,
                    "is_primary": term in primary_terms,
                    "term": term,
                    "score": score,
                    "payload_json": payload,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
        for rank, term in enumerate(_safe_list(block.get("document_primary_topics")), start=1):
            text = str(term or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "topic_id": _stable_hash_id("topic", run_id, document_id, block_name, "document_primary", rank, text),
                    "document_id": document_id,
                    "run_id": run_id,
                    "pipeline_profile": normalized_profile,
                    "topic_scope": "document_primary",
                    "topic_source": normalized_profile,
                    "page_index": None,
                    "sent_index": None,
                    "topic_rank": rank,
                    "is_primary": True,
                    "term": text,
                    "score": None,
                    "payload_json": {"term": text},
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
        for chunk_ordinal, chunk in enumerate(_safe_list(block.get("chunks_embeddings")), start=1):
            if not isinstance(chunk, dict):
                continue
            chunk_page = _coerce_int(chunk.get("page_index"))
            chunk_sent = _coerce_int(chunk.get("sent_index"))
            primary = str(chunk.get("chunk_primary_topic") or "").strip()
            if primary:
                rows.append(
                    {
                        "topic_id": _stable_hash_id("topic", run_id, document_id, block_name, "chunk_primary", chunk_page or "", chunk_sent or "", primary),
                        "document_id": document_id,
                        "run_id": run_id,
                        "pipeline_profile": normalized_profile,
                        "topic_scope": "chunk_primary",
                        "topic_source": normalized_profile,
                        "page_index": chunk_page,
                        "sent_index": chunk_sent,
                        "topic_rank": 1,
                        "is_primary": True,
                        "term": primary,
                        "score": None,
                        "payload_json": {"chunk_index": chunk_ordinal, "term": primary},
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    }
                )
            for rank, topic in enumerate(_safe_list(chunk.get("chunk_topics")), start=1):
                if isinstance(topic, dict):
                    term = str(topic.get("term") or "").strip()
                    score = _coerce_float(topic.get("score"))
                    payload = topic
                else:
                    term = str(topic or "").strip()
                    score = None
                    payload = {"term": topic}
                if not term:
                    continue
                rows.append(
                    {
                        "topic_id": _stable_hash_id("topic", run_id, document_id, block_name, "chunk", chunk_page or "", chunk_sent or "", rank, term),
                        "document_id": document_id,
                        "run_id": run_id,
                        "pipeline_profile": normalized_profile,
                        "topic_scope": "chunk",
                        "topic_source": normalized_profile,
                        "page_index": chunk_page,
                        "sent_index": chunk_sent,
                        "topic_rank": rank,
                        "is_primary": primary == term if primary else (rank == 1),
                        "term": term,
                        "score": score,
                        "payload_json": payload,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    }
                )
    return rows


def _build_document_sentence_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    nlp = _safe_dict(doc.get("nlp"))
    rows: List[Dict[str, Any]] = []
    for ordinal, sentence in enumerate(_safe_list(nlp.get("sentences")), start=1):
        if not isinstance(sentence, dict):
            continue
        page_index = _coerce_int(sentence.get("page_index"))
        sent_index = _coerce_int(_pick_first_non_empty(sentence.get("sent_index"), ordinal - 1))
        text = str(sentence.get("text") or "").strip()
        rows.append(
            {
                "sentence_id": _stable_hash_id("sentence", run_id, document_id, page_index or "", sent_index or "", text),
                "document_id": document_id,
                "run_id": run_id,
                "page_index": page_index,
                "sent_index": sent_index,
                "char_start": _coerce_int(_pick_first_non_empty(sentence.get("char_start"), sentence.get("start"))),
                "char_end": _coerce_int(_pick_first_non_empty(sentence.get("char_end"), sentence.get("end"))),
                "lang": str(sentence.get("lang") or nlp.get("language") or "").strip() or None,
                "text": text or None,
                "text_normalized": str(sentence.get("text_normalized") or "").strip() or None,
                "tokens_count": _coerce_int(_pick_first_non_empty(sentence.get("tokens_count"), sentence.get("token_count"))),
                "source_location_json": _json_or_none(
                    sentence.get("source_location")
                    or {
                        "page_index": page_index,
                        "sent_index": sent_index,
                        "char_start": _coerce_int(_pick_first_non_empty(sentence.get("char_start"), sentence.get("start"))),
                        "char_end": _coerce_int(_pick_first_non_empty(sentence.get("char_end"), sentence.get("end"))),
                    }
                ),
                "payload_json": sentence,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_document_token_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    nlp = _safe_dict(doc.get("nlp"))
    rows: List[Dict[str, Any]] = []
    for ordinal, token in enumerate(_safe_list(nlp.get("tokens")), start=1):
        if not isinstance(token, dict):
            continue
        page_index = _coerce_int(token.get("page_index"))
        sent_index = _coerce_int(token.get("sent_index"))
        tok_index = _coerce_int(_pick_first_non_empty(token.get("tok_index"), ordinal - 1))
        token_text = str(token.get("token") or "").strip()
        rows.append(
            {
                "token_id": _stable_hash_id("token", run_id, document_id, page_index or "", sent_index or "", tok_index or "", token_text),
                "document_id": document_id,
                "run_id": run_id,
                "page_index": page_index,
                "sent_index": sent_index,
                "tok_index": tok_index,
                "char_start": _coerce_int(_pick_first_non_empty(token.get("char_start"), token.get("start"))),
                "char_end": _coerce_int(_pick_first_non_empty(token.get("char_end"), token.get("end"))),
                "lang": str(token.get("lang") or nlp.get("language") or "").strip() or None,
                "token": token_text or None,
                "lemma": str(token.get("lemma") or "").strip() or None,
                "pos": str(token.get("pos") or "").strip() or None,
                "ner": str(token.get("ner") or "").strip() or None,
                "xlmr_backend": str(token.get("xlmr_backend") or "").strip() or None,
                "source_location_json": _json_or_none(
                    token.get("source_location")
                    or {
                        "page_index": page_index,
                        "sent_index": sent_index,
                        "tok_index": tok_index,
                        "char_start": _coerce_int(_pick_first_non_empty(token.get("char_start"), token.get("start"))),
                        "char_end": _coerce_int(_pick_first_non_empty(token.get("char_end"), token.get("end"))),
                    }
                ),
                "payload_json": token,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_document_entity_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    nlp = _safe_dict(doc.get("nlp"))
    rows: List[Dict[str, Any]] = []
    for ordinal, entity in enumerate(_safe_list(nlp.get("entities")), start=1):
        if not isinstance(entity, dict):
            continue
        page_index = _coerce_int(entity.get("page_index"))
        sent_index = _coerce_int(entity.get("sent_index"))
        text = str(entity.get("text") or "").strip()
        rows.append(
            {
                "entity_id": _stable_hash_id("entity", run_id, document_id, page_index or "", sent_index or "", entity.get("type") or "", text),
                "document_id": document_id,
                "run_id": run_id,
                "page_index": page_index,
                "sent_index": sent_index,
                "char_start": _coerce_int(_pick_first_non_empty(entity.get("char_start"), entity.get("start"))),
                "char_end": _coerce_int(_pick_first_non_empty(entity.get("char_end"), entity.get("end"))),
                "lang": str(entity.get("lang") or nlp.get("language") or "").strip() or None,
                "entity_type": str(entity.get("type") or entity.get("entity_type") or "").strip() or None,
                "text": text or None,
                "text_normalized": str(entity.get("text_normalized") or entity.get("normalized_text") or "").strip() or None,
                "canonical_text": str(entity.get("canonical_text") or entity.get("canonical") or "").strip() or None,
                "source_location_json": _json_or_none(
                    entity.get("source_location")
                    or {
                        "page_index": page_index,
                        "sent_index": sent_index,
                        "char_start": _coerce_int(_pick_first_non_empty(entity.get("char_start"), entity.get("start"))),
                        "char_end": _coerce_int(_pick_first_non_empty(entity.get("char_end"), entity.get("end"))),
                    }
                ),
                "payload_json": entity,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_document_nlp_match_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    nlp = _safe_dict(doc.get("nlp"))
    rows: List[Dict[str, Any]] = []
    for ordinal, match in enumerate(_safe_list(nlp.get("matches")), start=1):
        if not isinstance(match, dict):
            continue
        rows.append(
            {
                "nlp_match_id": _stable_hash_id("nlpmatch", run_id, document_id, ordinal, match.get("kind") or match.get("type") or "", _extract_item_text_excerpt(match) or ""),
                "document_id": document_id,
                "run_id": run_id,
                "page_index": _coerce_int(match.get("page_index")),
                "sent_index": _coerce_int(match.get("sent_index")),
                "char_start": _coerce_int(_pick_first_non_empty(match.get("char_start"), match.get("start"))),
                "char_end": _coerce_int(_pick_first_non_empty(match.get("char_end"), match.get("end"))),
                "match_kind": str(match.get("kind") or match.get("type") or "").strip() or None,
                "score": _coerce_float(match.get("score")),
                "text_excerpt": _extract_item_text_excerpt(match),
                "source_location_json": _json_or_none(match.get("source_location")),
                "shared_terms_json": _json_or_none(match.get("shared_terms")),
                "shared_topics_json": _json_or_none(match.get("shared_topics")),
                "phrase_a_json": _json_or_none(match.get("phrase_a")),
                "phrase_b_json": _json_or_none(match.get("phrase_b")),
                "chunk_a_json": _json_or_none(match.get("chunk_a")),
                "chunk_b_json": _json_or_none(match.get("chunk_b")),
                "payload_json": match,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _vector_dim_from_payload(value: Any) -> Optional[int]:
    if isinstance(value, list) and value:
        return len(value)
    return None


def _build_document_vector_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    profile: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    normalized_profile = _normalize_pipeline_profile(profile)
    for block_name in _active_ml_block_names(normalized_profile, doc):
        block = _safe_dict(doc.get(block_name))
        vector = block.get("document_vector")
        if not block or not isinstance(vector, list) or not vector:
            continue
        rows.append(
            {
                "vector_id": _stable_hash_id("docvec", run_id, document_id, block_name),
                "document_id": document_id,
                "run_id": run_id,
                "pipeline_profile": normalized_profile,
                "vector_scope": "document",
                "method": str(block.get("embedding_method") or "").strip() or None,
                "model": str(block.get("model") or "").strip() or None,
                "vector_dim": _coerce_int(block.get("vector_dim")) or _vector_dim_from_payload(vector),
                "vector_json": vector,
                "payload_json": {
                    "document_primary_topics": block.get("document_primary_topics"),
                    "document_topics": block.get("document_topics"),
                    "chunk_count": block.get("chunk_count"),
                },
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    return rows


def _build_document_chunk_embedding_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    profile: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    normalized_profile = _normalize_pipeline_profile(profile)
    for block_name in _active_ml_block_names(normalized_profile, doc):
        block = _safe_dict(doc.get(block_name))
        for ordinal, chunk in enumerate(_safe_list(block.get("chunks_embeddings")), start=1):
            if not isinstance(chunk, dict):
                continue
            vector = chunk.get("vector")
            rows.append(
                {
                    "chunk_embedding_id": _stable_hash_id("chunkvec", run_id, document_id, block_name, chunk.get("page_index") or "", chunk.get("sent_index") or "", ordinal),
                    "document_id": document_id,
                    "run_id": run_id,
                    "pipeline_profile": normalized_profile,
                    "page_index": _coerce_int(chunk.get("page_index")),
                    "sent_index": _coerce_int(chunk.get("sent_index")),
                    "lang": str(chunk.get("lang") or "").strip() or None,
                    "token_count": _coerce_int(chunk.get("token_count")),
                    "text_preview": _text_excerpt(chunk.get("text_preview"), 1000),
                    "chunk_primary_topic": str(chunk.get("chunk_primary_topic") or "").strip() or None,
                    "chunk_topics_json": _json_or_none(chunk.get("chunk_topics")),
                    "vector_dim": _coerce_int(block.get("vector_dim")) or _vector_dim_from_payload(vector),
                    "vector_json": _json_or_none(vector),
                    "payload_json": chunk,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
    return rows


def _build_document_word_embedding_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    profile: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    normalized_profile = _normalize_pipeline_profile(profile)
    for block_name in _active_ml_block_names(normalized_profile, doc):
        block = _safe_dict(doc.get(block_name))
        for ordinal, word in enumerate(_safe_list(block.get("word_embeddings")), start=1):
            if not isinstance(word, dict):
                continue
            vector = word.get("vector")
            rows.append(
                {
                    "word_embedding_id": _stable_hash_id("wordvec", run_id, document_id, block_name, word.get("page_index") or "", word.get("sent_index") or "", word.get("tok_index") or "", ordinal),
                    "document_id": document_id,
                    "run_id": run_id,
                    "pipeline_profile": normalized_profile,
                    "page_index": _coerce_int(word.get("page_index")),
                    "sent_index": _coerce_int(word.get("sent_index")),
                    "tok_index": _coerce_int(word.get("tok_index")),
                    "lang": str(word.get("lang") or "").strip() or None,
                    "token": str(word.get("token") or "").strip() or None,
                    "lemma": str(word.get("lemma") or "").strip() or None,
                    "vector_dim": _coerce_int(block.get("vector_dim")) or _vector_dim_from_payload(vector),
                    "vector_json": _json_or_none(vector),
                    "payload_json": word,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
    return rows


def _row_signature(row: Dict[str, Any]) -> Tuple[str, ...]:
    return (
        str(row.get("table_index") or ""),
        str(row.get("page_index") or ""),
        str(row.get("row_index") or ""),
        str(row.get("reference") or ""),
        str(row.get("product") or ""),
        str(row.get("quantity") or ""),
        str(row.get("unit_price") or ""),
        str(row.get("total_ht") or ""),
        str(row.get("total_ttc") or ""),
        str(row.get("total") or ""),
    )


def _table_row_to_sql_row(
    row: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    document_table_id: Optional[str],
    source_block: str,
    fallback_page_index: Optional[int],
    fallback_table_index: Optional[int],
    ordinal: int,
    now_iso: str,
) -> Dict[str, Any]:
    table_index = _coerce_int(_pick_first_non_empty(row.get("table_index"), fallback_table_index))
    page_index = _coerce_int(_pick_first_non_empty(row.get("page_index"), fallback_page_index))
    row_index = _coerce_int(_pick_first_non_empty(row.get("row_index"), ordinal))
    row_id = _stable_hash_id(
        "dtable-row",
        run_id,
        document_id,
        source_block,
        document_table_id or "",
        table_index or "",
        page_index or "",
        row_index or ordinal,
        row.get("reference") or "",
        row.get("product") or "",
    )
    return {
        "table_row_id": row_id,
        "document_id": document_id,
        "run_id": run_id,
        "document_table_id": document_table_id,
        "source_block": source_block,
        "table_index": table_index,
        "page_index": page_index,
        "row_index": row_index,
        "reference": str(row.get("reference") or "").strip() or None,
        "product": str(row.get("product") or "").strip() or None,
        "description": str(row.get("description") or row.get("label") or "").strip() or None,
        "quantity": _coerce_float(row.get("quantity")),
        "unit_price": _coerce_float(row.get("unit_price")),
        "total_ht": _coerce_float(row.get("total_ht")),
        "total_ttc": _coerce_float(row.get("total_ttc")),
        "total": _coerce_float(row.get("total")),
        "computed_total": _coerce_float(row.get("computed_total")),
        "effective_total": _coerce_float(row.get("effective_total")),
        "effective_total_source": str(row.get("effective_total_source") or "").strip() or None,
        "difference": _coerce_float(row.get("difference")),
        "status": str(row.get("status") or "").strip() or None,
        "confidence": _coerce_float(row.get("confidence")),
        "raw_cells_json": _json_or_none(row.get("raw_cells")),
        "raw_row_json": row,
        "created_at": now_iso,
        "updated_at": now_iso,
    }


def _build_document_table_rows(doc: Dict[str, Any], *, run_id: str, document_id: str, now_iso: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    structure = _safe_dict(doc.get("document_structure"))
    extraction = _safe_dict(doc.get("extraction"))
    table_extraction = _safe_dict(extraction.get("table_extraction"))

    table_rows: List[Dict[str, Any]] = []
    row_rows: List[Dict[str, Any]] = []

    def add_table_source(source_block: str, tables: List[Any], detected_columns: Any = None, totals: Any = None) -> None:
        for ordinal, table in enumerate(tables, start=1):
            if not isinstance(table, dict):
                continue
            table_index = _coerce_int(_pick_first_non_empty(table.get("table_index"), ordinal))
            page_index = _coerce_int(table.get("page_index"))
            document_table_id = _stable_hash_id(
                "dtable",
                run_id,
                document_id,
                source_block,
                table_index or ordinal,
                page_index or "",
                table.get("table_type") or "",
            )
            table_row = {
                "document_table_id": document_table_id,
                "document_id": document_id,
                "run_id": run_id,
                "source_block": source_block,
                "table_index": table_index,
                "page_index": page_index,
                "table_type": str(table.get("table_type") or "").strip() or None,
                "header_map_json": _json_or_none(table.get("header_map")),
                "header_score": _coerce_float(table.get("header_score")),
                "rows_count": _coerce_int(_pick_first_non_empty(table.get("rows_count"), len(_safe_list(table.get("rows"))) or None)),
                "detected_columns_json": _json_or_none(detected_columns),
                "totals_json": _json_or_none(totals),
                "shape_json": _json_or_none(table.get("shape")),
                "raw_table_json": table,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            table_rows.append(table_row)
            for row_ordinal, row in enumerate(_safe_list(table.get("rows")), start=1):
                if not isinstance(row, dict):
                    row = {"raw_cells": row}
                row_rows.append(
                    _table_row_to_sql_row(
                        row,
                        run_id=run_id,
                        document_id=document_id,
                        document_table_id=document_table_id,
                        source_block=source_block,
                        fallback_page_index=page_index,
                        fallback_table_index=table_index,
                        ordinal=row_ordinal,
                        now_iso=now_iso,
                    )
                )

    add_table_source("document_structure.tables", _safe_list(structure.get("tables")))
    add_table_source(
        "extraction.table_extraction.tables",
        _safe_list(table_extraction.get("tables")),
        detected_columns=table_extraction.get("detected_columns"),
        totals=table_extraction.get("totals"),
    )

    extraction_table_map: Dict[Tuple[int, int], str] = {}
    for table in table_rows:
        if table.get("source_block") != "extraction.table_extraction.tables":
            continue
        key = (_coerce_int(table.get("table_index")) or 0, _coerce_int(table.get("page_index")) or 0)
        extraction_table_map[key] = str(table["document_table_id"])

    seen = {_row_signature(row) for row in row_rows if isinstance(row, dict)}
    for ordinal, row in enumerate(_safe_list(table_extraction.get("line_items")), start=1):
        if not isinstance(row, dict):
            continue
        sig = _row_signature(row)
        if sig in seen:
            continue
        seen.add(sig)
        table_index = _coerce_int(row.get("table_index"))
        page_index = _coerce_int(row.get("page_index"))
        parent_id = extraction_table_map.get((table_index or 0, page_index or 0))
        row_rows.append(
            _table_row_to_sql_row(
                row,
                run_id=run_id,
                document_id=document_id,
                document_table_id=parent_id,
                source_block="extraction.table_extraction.line_items",
                fallback_page_index=page_index,
                fallback_table_index=table_index,
                ordinal=ordinal,
                now_iso=now_iso,
            )
        )

    _merge_table_row_quality_audit(doc, row_rows)
    return table_rows, row_rows


def _build_document_table_cell_rows(
    row_rows: List[Dict[str, Any]],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in row_rows:
        if not isinstance(row, dict):
            continue
        source_block = str(row.get("source_block") or "table").strip() or "table"
        table_index = _coerce_int(row.get("table_index"))
        page_index = _coerce_int(row.get("page_index"))
        row_index = _coerce_int(row.get("row_index"))
        row_id = str(row.get("table_row_id") or "").strip() or None
        table_id = str(row.get("document_table_id") or "").strip() or None
        raw_cells = row.get("raw_cells_json") if isinstance(row.get("raw_cells_json"), list) else []
        named_values = {
            key: str(value).strip()
            for key, value in (
                ("reference", row.get("reference")),
                ("product", row.get("product")),
                ("description", row.get("description")),
                ("quantity", row.get("quantity")),
                ("unit_price", row.get("unit_price")),
                ("total_ht", row.get("total_ht")),
                ("total_ttc", row.get("total_ttc")),
                ("total", row.get("total")),
            )
            if value not in (None, "") and str(value).strip()
        }

        cells: List[Dict[str, Any]] = []
        if raw_cells:
            for col_index, raw in enumerate(raw_cells, start=1):
                if isinstance(raw, dict):
                    raw_text = str(raw.get("text") or raw.get("value") or "").strip()
                    payload = raw
                else:
                    raw_text = str(raw or "").strip()
                    payload = {"value": raw}
                inferred_name = None
                for name, value_text in named_values.items():
                    if raw_text and raw_text == value_text:
                        inferred_name = name
                        break
                if not raw_text and not payload:
                    continue
                cells.append(
                    {
                        "col_index": col_index,
                        "column_name": str(payload.get("column_name") or inferred_name or "").strip() or None,
                        "header_path": str(payload.get("header_path") or inferred_name or "").strip() or None,
                        "cell_role": str(payload.get("cell_role") or inferred_name or "").strip() or None,
                        "raw_text": raw_text or None,
                        "normalized_text": str(payload.get("normalized_text") or raw_text or "").strip() or None,
                        "numeric_value": _coerce_float(payload.get("numeric_value") if isinstance(payload, dict) else raw_text),
                        "value_type": str(payload.get("value_type") or ("number" if _coerce_float(raw_text) is not None else "text")).strip() or None,
                        "unit_text": str(payload.get("unit_text") or "").strip() or None,
                        "currency_text": str(payload.get("currency_text") or "").strip() or None,
                        "confidence": _coerce_float(payload.get("confidence") if isinstance(payload, dict) else row.get("confidence")),
                        "payload_json": payload,
                    }
                )
        else:
            ordered_fields = [
                ("reference", row.get("reference")),
                ("product", row.get("product")),
                ("description", row.get("description")),
                ("quantity", row.get("quantity")),
                ("unit_price", row.get("unit_price")),
                ("total_ht", row.get("total_ht")),
                ("total_ttc", row.get("total_ttc")),
                ("total", row.get("total")),
            ]
            for col_index, (column_name, value) in enumerate(ordered_fields, start=1):
                if value in (None, ""):
                    continue
                raw_text = str(value).strip()
                cells.append(
                    {
                        "col_index": col_index,
                        "column_name": column_name,
                        "header_path": column_name,
                        "cell_role": column_name,
                        "raw_text": raw_text or None,
                        "normalized_text": raw_text or None,
                        "numeric_value": _coerce_float(value),
                        "value_type": "number" if _coerce_float(value) is not None else "text",
                        "unit_text": None,
                        "currency_text": None,
                        "confidence": _coerce_float(row.get("confidence")),
                        "payload_json": {"column_name": column_name, "value": value},
                    }
                )

        for cell in cells:
            col_index = _coerce_int(cell.get("col_index"))
            out.append(
                {
                    "table_cell_id": _stable_hash_id("dtable-cell", run_id, document_id, table_id or "", row_id or "", row_index or "", col_index or "", cell.get("raw_text") or ""),
                    "document_id": document_id,
                    "run_id": run_id,
                    "document_table_id": table_id,
                    "table_row_id": row_id,
                    "source_block": source_block,
                    "table_index": table_index,
                    "page_index": page_index,
                    "row_index": row_index,
                    "col_index": col_index,
                    "column_name": str(cell.get("column_name") or "").strip() or None,
                    "header_path": str(cell.get("header_path") or "").strip() or None,
                    "cell_role": str(cell.get("cell_role") or "").strip() or None,
                    "raw_text": str(cell.get("raw_text") or "").strip() or None,
                    "normalized_text": str(cell.get("normalized_text") or "").strip() or None,
                    "numeric_value": _coerce_float(cell.get("numeric_value")),
                    "value_type": str(cell.get("value_type") or "").strip() or None,
                    "unit_text": str(cell.get("unit_text") or "").strip() or None,
                    "currency_text": str(cell.get("currency_text") or "").strip() or None,
                    "confidence": _coerce_float(cell.get("confidence")),
                    "payload_json": _json_or_none(cell.get("payload_json")),
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
    return out


def _build_quality_check_row(
    row: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    source_block: str,
    ordinal: int,
    now_iso: str,
) -> Dict[str, Any]:
    check_name = str(row.get("check") or row.get("name") or row.get("verification_status") or source_block).strip() or source_block
    quality_check_id = _stable_hash_id("qcheck", run_id, document_id, source_block, ordinal, check_name)
    return {
        "quality_check_id": quality_check_id,
        "document_id": document_id,
        "run_id": run_id,
        "source_block": source_block,
        "check_name": check_name,
        "engine": str(row.get("engine") or "").strip() or None,
        "status": str(row.get("status") or row.get("verification_status") or "").strip() or None,
        "passed": _optional_bool(_pick_first_non_empty(row.get("passed"), row.get("ok"))),
        "complete": _optional_bool(row.get("complete")),
        "rows_total": _coerce_int(row.get("rows_total")),
        "row_ok_count": _coerce_int(row.get("row_ok_count")),
        "row_partial_count": _coerce_int(row.get("row_partial_count")),
        "row_mismatch_count": _coerce_int(row.get("row_mismatch_count")),
        "rows_verified": _coerce_int(row.get("rows_verified")),
        "computed_subtotal": _coerce_float(row.get("computed_subtotal")),
        "declared_subtotal": _coerce_float(row.get("declared_subtotal")),
        "declared_tax": _coerce_float(row.get("declared_tax")),
        "computed_tax": _coerce_float(row.get("computed_tax")),
        "declared_total": _coerce_float(row.get("declared_total")),
        "expected_total": _coerce_float(row.get("expected_total")),
        "subtotal_status": str(row.get("subtotal_status") or "").strip() or None,
        "tax_status": str(row.get("tax_status") or "").strip() or None,
        "total_status": str(row.get("total_status") or "").strip() or None,
        "declared_totals_raw_json": _json_or_none(row.get("declared_totals_raw")),
        "tolerance": str(row.get("tolerance") or "").strip() or None,
        "table_anchor_json": _json_or_none(row.get("table_anchor")),
        "subtotal_location_json": _json_or_none(row.get("subtotal_location")),
        "tax_location_json": _json_or_none(row.get("tax_location")),
        "total_location_json": _json_or_none(row.get("total_location")),
        "issue_locations_json": _json_or_none(row.get("issue_locations")),
        "details_json": _json_or_none(row.get("details") or row.get("row_audit") or row.get("checks")),
        "raw_check_json": row,
        "created_at": now_iso,
        "updated_at": now_iso,
    }


def _iter_quality_rows(doc: Dict[str, Any]) -> List[Tuple[str, int, Dict[str, Any]]]:
    out: List[Tuple[str, int, Dict[str, Any]]] = []
    for ordinal, row in enumerate(_safe_list(doc.get("quality_checks")), start=1):
        if isinstance(row, dict):
            out.append(("quality_checks", ordinal, row))
    extraction = _safe_dict(doc.get("extraction"))
    for ordinal, row in enumerate(_safe_list(extraction.get("quality_checks")), start=1):
        if isinstance(row, dict):
            out.append(("extraction.quality_checks", ordinal, row))
    totals_verification = _safe_dict(extraction.get("totals_verification"))
    if totals_verification:
        out.append(("extraction.totals_verification", 1, totals_verification))
    return out


def _merge_table_row_quality_audit(doc: Dict[str, Any], row_rows: List[Dict[str, Any]]) -> None:
    audit_by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for _, _, quality_row in _iter_quality_rows(doc):
        for audit in _safe_list(quality_row.get("row_audit")):
            if not isinstance(audit, dict):
                continue
            key = (
                _coerce_int(audit.get("table_index")) or 0,
                _coerce_int(audit.get("page_index")) or 0,
                _coerce_int(audit.get("row_index")) or 0,
            )
            audit_by_key[key] = audit

    for row in row_rows:
        key = (
            _coerce_int(row.get("table_index")) or 0,
            _coerce_int(row.get("page_index")) or 0,
            _coerce_int(row.get("row_index")) or 0,
        )
        audit = audit_by_key.get(key)
        if not audit:
            continue
        row["description"] = row.get("description") or str(audit.get("label") or "").strip() or None
        for field in ("computed_total", "effective_total", "difference"):
            if row.get(field) is None:
                row[field] = _coerce_float(audit.get(field))
        if row.get("effective_total_source") is None:
            row["effective_total_source"] = str(audit.get("effective_total_source") or "").strip() or None
        if row.get("status") is None:
            row["status"] = str(audit.get("status") or "").strip() or None


def _build_document_quality_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    quality_rows: List[Dict[str, Any]] = []
    issue_rows: List[Dict[str, Any]] = []
    row_audit_rows: List[Dict[str, Any]] = []
    declared_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    for source_block, ordinal, row in _iter_quality_rows(doc):
        quality_row = _build_quality_check_row(
            row,
            run_id=run_id,
            document_id=document_id,
            source_block=source_block,
            ordinal=ordinal,
            now_iso=now_iso,
        )
        quality_rows.append(quality_row)
        quality_check_id = quality_row["quality_check_id"]

        for issue_index, issue in enumerate(_safe_list(row.get("issue_locations")), start=1):
            if not isinstance(issue, dict):
                continue
            issue_rows.append(
                {
                    "issue_id": _stable_hash_id("qissue", run_id, document_id, quality_check_id, issue_index, issue.get("kind") or ""),
                    "quality_check_id": quality_check_id,
                    "document_id": document_id,
                    "run_id": run_id,
                    "kind": str(issue.get("kind") or "").strip() or None,
                    "table_index": _coerce_int(issue.get("table_index")),
                    "page_index": _coerce_int(issue.get("page_index")),
                    "row_index": _coerce_int(issue.get("row_index")),
                    "computed": _coerce_float(issue.get("computed")),
                    "declared": _coerce_float(issue.get("declared")),
                    "source_location_json": _json_or_none(issue.get("source_location")),
                    "payload_json": issue,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )

        for audit_index, audit in enumerate(_safe_list(row.get("row_audit")), start=1):
            if not isinstance(audit, dict):
                continue
            row_audit_rows.append(
                {
                    "row_audit_id": _stable_hash_id("qrow", run_id, document_id, quality_check_id, audit_index, audit.get("table_index") or "", audit.get("row_index") or ""),
                    "quality_check_id": quality_check_id,
                    "document_id": document_id,
                    "run_id": run_id,
                    "table_index": _coerce_int(audit.get("table_index")),
                    "page_index": _coerce_int(audit.get("page_index")),
                    "row_index": _coerce_int(audit.get("row_index")),
                    "quantity": _coerce_float(audit.get("quantity")),
                    "unit_price": _coerce_float(audit.get("unit_price")),
                    "computed_total": _coerce_float(audit.get("computed_total")),
                    "effective_total": _coerce_float(audit.get("effective_total")),
                    "effective_total_source": str(audit.get("effective_total_source") or "").strip() or None,
                    "difference": _coerce_float(audit.get("difference")),
                    "status": str(audit.get("status") or "").strip() or None,
                    "source_location_json": _json_or_none(audit.get("source_location")),
                    "payload_json": audit,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )

        for location_kind in ("table_anchor", "subtotal_location", "tax_location", "total_location"):
            location = _safe_dict(row.get(location_kind))
            if not location:
                continue
            declared_rows.append(
                {
                    "declared_location_id": _stable_hash_id("qdecl", run_id, document_id, quality_check_id, location_kind),
                    "quality_check_id": quality_check_id,
                    "document_id": document_id,
                    "run_id": run_id,
                    "location_kind": location_kind,
                    "page_index": _coerce_int(location.get("page_index")),
                    "table_index": _coerce_int(location.get("table_index")),
                    "source_location_json": _json_or_none(location),
                    "payload_json": location,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )

        for step_index, step in enumerate(_safe_list(row.get("checks")), start=1):
            if not isinstance(step, dict):
                continue
            step_rows.append(
                {
                    "check_step_id": _stable_hash_id("qstep", run_id, document_id, quality_check_id, step_index, step.get("name") or ""),
                    "quality_check_id": quality_check_id,
                    "document_id": document_id,
                    "run_id": run_id,
                    "step_index": step_index,
                    "step_name": str(step.get("name") or "").strip() or None,
                    "status": str(step.get("status") or "").strip() or None,
                    "payload_json": step,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )

    return quality_rows, issue_rows, row_audit_rows, declared_rows, step_rows


def _feature_payload_summary(payload: Dict[str, Any], keys: List[str]) -> Optional[Dict[str, Any]]:
    out = {key: payload.get(key) for key in keys if key in payload}
    return out or None


def _build_document_feature_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    profile: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def add_feature(
        feature_group: str,
        feature_name: str,
        *,
        backend: Any = None,
        method: Any = None,
        model: Any = None,
        vector_dim: Any = None,
        count_value: Any = None,
        score_value: Any = None,
        text_value: Any = None,
        bool_value: Any = None,
        topics_json: Any = None,
        payload_json: Any = None,
    ) -> None:
        feature_id = _stable_hash_id("feature", run_id, document_id, feature_group, feature_name)
        out.append(
            {
                "feature_id": feature_id,
                "document_id": document_id,
                "run_id": run_id,
                "pipeline_profile": profile,
                "feature_group": feature_group,
                "feature_name": feature_name,
                "backend": str(backend).strip() or None if backend is not None else None,
                "method": str(method).strip() or None if method is not None else None,
                "model": str(model).strip() or None if model is not None else None,
                "vector_dim": _coerce_int(vector_dim),
                "count_value": _coerce_int(count_value),
                "score_value": _coerce_float(score_value),
                "text_value": str(text_value).strip() or None if text_value is not None else None,
                "bool_value": _optional_bool(bool_value),
                "topics_json": _json_or_none(topics_json),
                "payload_json": _json_or_none(payload_json),
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    classification = _safe_dict(doc.get("classification"))
    if classification:
        add_feature(
            "classification",
            "classification",
            method="rules-score",
            count_value=len(_safe_dict(classification.get("scores"))) or None,
            score_value=classification.get("winning_score"),
            text_value=classification.get("doc_type") or classification.get("status"),
            payload_json=_feature_payload_summary(
                classification,
                ["doc_type", "status", "winning_score", "threshold", "margin", "scores", "keyword_matches", "anti_confusion_targets"],
            ),
        )

    nlp = _safe_dict(doc.get("nlp"))
    if nlp:
        add_feature(
            "nlp",
            "nlp",
            method="pipeline-output",
            count_value=len(_safe_list(nlp.get("tokens"))) or len(_safe_list(nlp.get("sentences"))) or None,
            text_value=nlp.get("language"),
            payload_json={
                "language": nlp.get("language"),
                "sentences_count": len(_safe_list(nlp.get("sentences"))),
                "entities_count": len(_safe_list(nlp.get("entities"))),
                "matches_count": len(_safe_list(nlp.get("matches"))),
                "tokens_count": len(_safe_list(nlp.get("tokens"))),
            },
        )

    for block_name in _active_ml_block_names(profile, doc):
        block = _safe_dict(doc.get(block_name))
        if not block:
            continue
        add_feature(
            "embeddings",
            _normalize_pipeline_profile(profile),
            backend=_normalize_pipeline_profile(profile),
            method=block.get("embedding_method"),
            vector_dim=block.get("vector_dim"),
            count_value=block.get("chunk_count"),
            text_value=(block.get("document_primary_topics") or [None])[0],
            topics_json=block.get("document_topics"),
            payload_json={
                "document_primary_topics": block.get("document_primary_topics"),
                "document_topics": block.get("document_topics"),
                "chunk_count": block.get("chunk_count"),
                "word_embeddings_count": len(_safe_list(block.get("word_embeddings"))),
                "has_document_vector": bool(block.get("document_vector")),
            },
        )

    components = _safe_dict(doc.get("components"))
    grammar = _safe_dict(components.get("attribution_grammaticale"))
    if grammar:
        add_feature(
            "component",
            "attribution_grammaticale",
            backend=grammar.get("backend"),
            method=grammar.get("pos_method"),
            model=grammar.get("model"),
            count_value=grammar.get("entities_count") or grammar.get("sentences_count"),
            payload_json=grammar,
        )

    elasticsearch = _safe_dict(components.get("elasticsearch"))
    if elasticsearch:
        add_feature(
            "component",
            "elasticsearch",
            backend="elasticsearch",
            method=elasticsearch.get("nlp_level"),
            text_value=elasticsearch.get("es_index"),
            bool_value=elasticsearch.get("available"),
            payload_json=elasticsearch,
        )

    visual_flags = _extract_visual_flags(doc)
    if visual_flags:
        add_feature(
            "document",
            "visual_flags",
            method="fusion-output",
            bool_value=any(_optional_bool(value) for value in visual_flags.values()),
            payload_json=visual_flags,
        )

    cross_document = _safe_dict(doc.get("cross_document"))
    if cross_document:
        add_feature(
            "document",
            "cross_document",
            method="fusion-output",
            count_value=cross_document.get("linked_documents_count"),
            payload_json=cross_document,
        )

    return out


def _build_document_extraction_detail_rows(
    doc: Dict[str, Any],
    *,
    run_id: str,
    document_id: str,
    now_iso: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    extraction = _safe_dict(doc.get("extraction"))
    detail_rows: List[Dict[str, Any]] = []
    regex_field_rows: List[Dict[str, Any]] = []
    regex_match_rows: List[Dict[str, Any]] = []
    bm25_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []

    for regex_index, regex_row in enumerate(_safe_list(extraction.get("regex_extractions")), start=1):
        if not isinstance(regex_row, dict):
            continue
        ruleset = _safe_dict(regex_row.get("ruleset"))
        extractor_version = ",".join(str(x) for x in _safe_list(ruleset.get("applied_rulesets")) if str(x).strip()) or None
        fields = _safe_dict(regex_row.get("fields"))
        for field_name, field_payload in fields.items():
            if not isinstance(field_payload, dict):
                field_payload = {"value": field_payload}
            rule_id = str(field_payload.get("rule_id") or "").strip() or None
            field_type = str(field_payload.get("type") or "").strip() or None
            matches = [m for m in _safe_list(field_payload.get("matches")) if isinstance(m, dict)]
            extraction_id = _stable_hash_id("extract", run_id, document_id, regex_index, field_name, rule_id or "")
            detail_rows.append(
                {
                    "extraction_id": extraction_id,
                    "document_id": document_id,
                    "run_id": run_id,
                    "field_name": str(field_name),
                    "field_type": field_type,
                    "rule_id": rule_id,
                    "is_many": _optional_bool(field_payload.get("many")),
                    "value_text": str(matches[0].get("value") or "").strip() if len(matches) == 1 else None,
                    "value_json": {"matches": matches} if matches else field_payload,
                    "source_component": "extraction_regles",
                    "source_path": f"extraction.regex_extractions[{regex_index - 1}].fields.{field_name}",
                    "extractor_name": "regex-yaml",
                    "extractor_version": extractor_version,
                    "confidence_score": None,
                    "payload_json": field_payload,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            regex_field_rows.append(
                {
                    "regex_field_id": _stable_hash_id("regexfield", run_id, document_id, regex_index, field_name),
                    "extraction_id": extraction_id,
                    "document_id": document_id,
                    "run_id": run_id,
                    "field_name": str(field_name),
                    "field_type": field_type,
                    "rule_id": rule_id,
                    "is_many": _optional_bool(field_payload.get("many")),
                    "values_count": len(matches),
                    "payload_json": field_payload,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            regex_field_id = regex_field_rows[-1]["regex_field_id"]
            for match_index, match in enumerate(matches, start=1):
                regex_match_rows.append(
                    {
                        "regex_match_id": _stable_hash_id("regexmatch", run_id, document_id, regex_index, field_name, match_index, match.get("value") or ""),
                        "regex_field_id": regex_field_id,
                        "document_id": document_id,
                        "run_id": run_id,
                        "field_name": str(field_name),
                        "page_index": _coerce_int(match.get("page_index")),
                        "start_offset": _coerce_int(match.get("start")),
                        "end_offset": _coerce_int(match.get("end")),
                        "match_value": str(match.get("value") or "").strip() or None,
                        "snippet": _text_excerpt(match.get("snippet"), 1000),
                        "payload_json": match,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    }
                )
        bm25 = _safe_dict(regex_row.get("bm25"))
        for chunk_index, chunk in enumerate(_safe_list(bm25.get("top_chunks")), start=1):
            if not isinstance(chunk, dict):
                continue
            bm25_rows.append(
                {
                    "bm25_chunk_id": _stable_hash_id("bm25", run_id, document_id, regex_index, chunk_index, chunk.get("page_index") or "", chunk.get("sent_index") or ""),
                    "document_id": document_id,
                    "run_id": run_id,
                    "page_index": _coerce_int(chunk.get("page_index")),
                    "sent_index": _coerce_int(chunk.get("sent_index")),
                    "score": _coerce_float(chunk.get("score")),
                    "text_preview": _text_excerpt(chunk.get("text_preview"), 1000),
                    "payload_json": chunk,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )

    business = extraction.get("business")
    if isinstance(business, dict):
        for field_name, field_payload in business.items():
            detail_rows.append(
                {
                    "extraction_id": _stable_hash_id("extract", run_id, document_id, "business", field_name),
                    "document_id": document_id,
                    "run_id": run_id,
                    "field_name": str(field_name),
                    "field_type": str(_safe_dict(field_payload).get("type") or "business").strip() or "business",
                    "rule_id": None,
                    "is_many": isinstance(field_payload, list),
                    "value_text": str(field_payload).strip() if isinstance(field_payload, (str, int, float, bool)) else None,
                    "value_json": None if isinstance(field_payload, (str, int, float, bool)) else field_payload,
                    "source_component": "extraction_business",
                    "source_path": f"extraction.business.{field_name}",
                    "extractor_name": "business",
                    "extractor_version": None,
                    "confidence_score": _coerce_float(_safe_dict(field_payload).get("confidence")),
                    "payload_json": field_payload,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )

    for relation_index, relation in enumerate(_safe_list(extraction.get("relations")), start=1):
        if not isinstance(relation, dict):
            continue
        relation_rows.append(
            {
                "relation_id": _stable_hash_id("relation", run_id, document_id, relation_index, relation.get("type") or relation.get("relation_type") or ""),
                "document_id": document_id,
                "run_id": run_id,
                "relation_type": str(relation.get("type") or relation.get("relation_type") or "").strip() or None,
                "subject_text": str(relation.get("subject") or relation.get("subject_text") or "").strip() or None,
                "predicate_text": str(relation.get("predicate") or relation.get("predicate_text") or "").strip() or None,
                "object_text": str(relation.get("object") or relation.get("object_text") or "").strip() or None,
                "evidence_text": _text_excerpt(relation.get("evidence") or relation.get("evidence_text"), 1000),
                "source_path": f"extraction.relations[{relation_index - 1}]",
                "confidence_score": _coerce_float(relation.get("confidence") or relation.get("score")),
                "source_location_json": _json_or_none(relation.get("source_location")),
                "payload_json": relation,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    return detail_rows, regex_field_rows, regex_match_rows, bm25_rows, relation_rows


def _collect_shared_terms(link: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for match in _safe_list(_safe_dict(link.get("audit")).get("matches")):
        if not isinstance(match, dict):
            continue
        for item in _safe_list(match.get("shared_terms")):
            text = str(item or "").strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                out.append(text)
    return out


def _build_link_document_lookup(payload: Dict[str, Any]) -> Dict[str, str]:
    doc_lookup: Dict[str, str] = {}
    for idx, doc in enumerate(_payload_documents(payload), start=1):
        doc_id = _extract_document_id(doc, idx)
        filename = _extract_filename(doc)
        doc_lookup[doc_id] = doc_id
        if filename:
            doc_lookup[filename.lower()] = doc_id
    return doc_lookup


def _resolve_link_endpoint_id(endpoint: Dict[str, Any], doc_lookup: Dict[str, str]) -> Optional[str]:
    direct_id = str(endpoint.get("document_id") or "").strip()
    if direct_id:
        return direct_id
    filename = str(endpoint.get("filename") or "").strip().lower()
    return doc_lookup.get(filename)


def _build_document_link_shared_term_rows(
    payload: Dict[str, Any],
    *,
    run_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    links = _extract_links(payload)
    if not links:
        return []
    doc_lookup = _build_link_document_lookup(payload)
    rows: List[Dict[str, Any]] = []
    for ordinal, link in enumerate(links, start=1):
        doc_a = _safe_dict(link.get("doc_a"))
        doc_b = _safe_dict(link.get("doc_b"))
        source_document_id = _resolve_link_endpoint_id(doc_a, doc_lookup)
        target_document_id = _resolve_link_endpoint_id(doc_b, doc_lookup)
        if not source_document_id or not target_document_id or source_document_id == target_document_id:
            continue
        link_id = str(link.get("link_id") or "").strip() or _stable_hash_id(
            "link",
            run_id,
            source_document_id or doc_a.get("filename") or ordinal,
            target_document_id or doc_b.get("filename") or ordinal,
        )
        for term_rank, term in enumerate(_collect_shared_terms(link), start=1):
            examples_a, examples_b = _collect_link_term_examples(link, term)
            rows.append(
                {
                    "link_shared_term_id": _stable_hash_id("linkterm", run_id, link_id, term_rank, term),
                    "link_id": link_id,
                    "run_id": run_id,
                    "source_document_id": source_document_id,
                    "target_document_id": target_document_id,
                    "term_rank": term_rank,
                    "term": term,
                    "score": None,
                    "doc_a_examples_json": examples_a or None,
                    "doc_b_examples_json": examples_b or None,
                    "payload_json": {"term": term},
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
    return rows


def _build_document_link_rows(
    payload: Dict[str, Any],
    *,
    run_id: str,
    now_iso: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    links = _extract_links(payload)
    if not links:
        return [], [], [], []

    doc_lookup = _build_link_document_lookup(payload)

    out: List[Dict[str, Any]] = []
    shared_topic_rows: List[Dict[str, Any]] = []
    sentence_match_rows: List[Dict[str, Any]] = []
    chunk_match_rows: List[Dict[str, Any]] = []
    for ordinal, link in enumerate(links, start=1):
        doc_a = _safe_dict(link.get("doc_a"))
        doc_b = _safe_dict(link.get("doc_b"))
        source_document_id = _resolve_link_endpoint_id(doc_a, doc_lookup)
        target_document_id = _resolve_link_endpoint_id(doc_b, doc_lookup)
        if not source_document_id or not target_document_id or source_document_id == target_document_id:
            continue
        link_id = str(link.get("link_id") or "").strip() or _stable_hash_id(
            "link",
            run_id,
            source_document_id or doc_a.get("filename") or ordinal,
            target_document_id or doc_b.get("filename") or ordinal,
        )
        vector_audit = _safe_dict(link.get("vector_audit"))
        audit = _safe_dict(link.get("audit"))
        out.append(
            {
                "link_id": link_id,
                "run_id": run_id,
                "source_document_id": source_document_id or None,
                "target_document_id": target_document_id or None,
                "source_filename": str(doc_a.get("filename") or "").strip() or None,
                "target_filename": str(doc_b.get("filename") or "").strip() or None,
                "link_type": "cross_document",
                "score": _coerce_float(link.get("score")),
                "vector_profile": str(vector_audit.get("profile") or "").strip() or None,
                "embedding_method": str(vector_audit.get("embedding_method") or "").strip() or None,
                "embedding_backend": str(vector_audit.get("embedding_backend") or "").strip() or None,
                "vector_dim": _coerce_int(vector_audit.get("vector_dim")),
                "doc_similarity": _coerce_float(vector_audit.get("doc_similarity")),
                "sentence_matches_count": _coerce_int(audit.get("sentence_matches_count")),
                "chunk_matches_count": _coerce_int(vector_audit.get("chunk_matches_count")),
                "shared_topics_json": _json_or_none(link.get("shared_topics")),
                "shared_terms_json": _collect_shared_terms(link) or None,
                "score_breakdown_json": _json_or_none(link.get("score_breakdown")),
                "audit_json": _json_or_none(audit),
                "vector_audit_json": _json_or_none(vector_audit),
                "raw_link_json": link,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
        for topic_rank, topic in enumerate(_safe_list(link.get("shared_topics")), start=1):
            if not isinstance(topic, dict):
                continue
            shared_topic_rows.append(
                {
                    "link_shared_topic_id": _stable_hash_id("linktopic", run_id, link_id, topic_rank, topic.get("term") or ""),
                    "link_id": link_id,
                    "run_id": run_id,
                    "source_document_id": source_document_id or None,
                    "target_document_id": target_document_id or None,
                    "topic_rank": topic_rank,
                    "term": str(topic.get("term") or "").strip() or None,
                    "score": _coerce_float(topic.get("score")),
                    "doc_a_examples_json": _json_or_none(topic.get("doc_a_examples")),
                    "doc_b_examples_json": _json_or_none(topic.get("doc_b_examples")),
                    "payload_json": topic,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
        for match_index, match in enumerate(_safe_list(audit.get("matches")), start=1):
            if not isinstance(match, dict):
                continue
            sentence_match_rows.append(
                {
                    "link_sentence_match_id": _stable_hash_id("linksent", run_id, link_id, match_index),
                    "link_id": link_id,
                    "run_id": run_id,
                    "source_document_id": source_document_id or None,
                    "target_document_id": target_document_id or None,
                    "match_index": match_index,
                    "score": _coerce_float(match.get("score")),
                    "shared_terms_json": _json_or_none(match.get("shared_terms")),
                    "shared_topics_json": _json_or_none(match.get("shared_topics")),
                    "phrase_a_json": _json_or_none(match.get("phrase_a")),
                    "phrase_b_json": _json_or_none(match.get("phrase_b")),
                    "payload_json": match,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
        for match_index, match in enumerate(_safe_list(vector_audit.get("chunk_matches")), start=1):
            if not isinstance(match, dict):
                continue
            chunk_match_rows.append(
                {
                    "link_chunk_match_id": _stable_hash_id("linkchunk", run_id, link_id, match_index),
                    "link_id": link_id,
                    "run_id": run_id,
                    "source_document_id": source_document_id or None,
                    "target_document_id": target_document_id or None,
                    "match_index": match_index,
                    "score": _coerce_float(match.get("score")),
                    "vector_similarity": _coerce_float(match.get("vector_similarity")),
                    "shared_terms_json": _json_or_none(match.get("shared_terms")),
                    "shared_topics_json": _json_or_none(match.get("shared_topics")),
                    "chunk_a_json": _json_or_none(match.get("chunk_a")),
                    "chunk_b_json": _json_or_none(match.get("chunk_b")),
                    "payload_json": match,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
    return out, shared_topic_rows, sentence_match_rows, chunk_match_rows


def _build_stable_registry_rows(
    *,
    run_id: str,
    now_iso: str,
    document_row: Optional[Dict[str, Any]] = None,
    identifier_rows: Optional[List[Dict[str, Any]]] = None,
    entity_rows: Optional[List[Dict[str, Any]]] = None,
    topic_rows: Optional[List[Dict[str, Any]]] = None,
    table_rows: Optional[List[Dict[str, Any]]] = None,
    relation_rows: Optional[List[Dict[str, Any]]] = None,
    link_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: Optional[Dict[str, Any]]) -> None:
        if not isinstance(row, dict):
            return
        registry_id = str(row.get("registry_id") or "").strip()
        if not registry_id or registry_id in seen:
            return
        seen.add(registry_id)
        rows.append(row)

    if isinstance(document_row, dict):
        add(
            _make_registry_row(
                entity_kind="document",
                natural_key_parts=[document_row.get("source_document_key"), document_row.get("file_sha256"), document_row.get("content_sha256")],
                stable_id=document_row.get("source_document_key") or document_row.get("document_id"),
                run_id=run_id,
                now_iso=now_iso,
                payload_json={"document_id": document_row.get("document_id"), "file_name": document_row.get("file_name")},
            )
        )

    for row in identifier_rows or []:
        add(
            _make_registry_row(
                entity_kind="identifier",
                natural_key_parts=[row.get("identifier_type"), row.get("value_normalized") or row.get("value_text")],
                stable_id=_stable_hash_id("stable-identifier", row.get("identifier_type"), row.get("value_normalized") or row.get("value_text")),
                run_id=run_id,
                now_iso=now_iso,
                payload_json=row,
            )
        )

    for row in entity_rows or []:
        add(
            _make_registry_row(
                entity_kind="entity",
                natural_key_parts=[row.get("entity_type"), row.get("canonical_text") or row.get("text_normalized") or row.get("text")],
                stable_id=_stable_hash_id("stable-entity", row.get("entity_type"), row.get("canonical_text") or row.get("text_normalized") or row.get("text")),
                run_id=run_id,
                now_iso=now_iso,
                payload_json=row,
            )
        )

    for row in topic_rows or []:
        add(
            _make_registry_row(
                entity_kind="topic",
                natural_key_parts=[row.get("pipeline_profile"), row.get("topic_scope"), row.get("term")],
                stable_id=_stable_hash_id("stable-topic", row.get("pipeline_profile"), row.get("topic_scope"), row.get("term")),
                run_id=run_id,
                now_iso=now_iso,
                payload_json=row,
            )
        )

    for row in table_rows or []:
        add(
            _make_registry_row(
                entity_kind="table",
                natural_key_parts=[row.get("document_id"), row.get("source_block"), row.get("table_index"), row.get("page_index"), row.get("table_type")],
                stable_id=_stable_hash_id("stable-table", row.get("document_id"), row.get("source_block"), row.get("table_index"), row.get("page_index"), row.get("table_type")),
                run_id=run_id,
                now_iso=now_iso,
                payload_json=row,
            )
        )

    for row in relation_rows or []:
        add(
            _make_registry_row(
                entity_kind="relation",
                natural_key_parts=[row.get("relation_type"), row.get("subject_text"), row.get("predicate_text"), row.get("object_text")],
                stable_id=_stable_hash_id("stable-relation", row.get("relation_type"), row.get("subject_text"), row.get("predicate_text"), row.get("object_text")),
                run_id=run_id,
                now_iso=now_iso,
                payload_json=row,
            )
        )

    for row in link_rows or []:
        add(
            _make_registry_row(
                entity_kind="document_link",
                natural_key_parts=[row.get("source_document_id"), row.get("target_document_id"), row.get("link_type")],
                stable_id=_stable_hash_id("stable-doclink", row.get("source_document_id"), row.get("target_document_id"), row.get("link_type")),
                run_id=run_id,
                now_iso=now_iso,
                payload_json=row,
            )
        )

    return rows


def build_postgres_sync_audit(sync_status: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": bool(sync_status.get("enabled")),
        "sync_enabled": bool(sync_status.get("sync_enabled")),
        "sync_write_fusion_audit": bool(sync_status.get("sync_write_fusion_audit")),
        "ready": bool(sync_status.get("ready")),
        "host": sync_status.get("host"),
        "port": sync_status.get("port"),
        "database": sync_status.get("database"),
        "run_id": sync_status.get("run_id"),
        "pipeline_profile": sync_status.get("pipeline_profile"),
        "source": sync_status.get("source"),
        "log_entries_created": int(sync_status.get("log_entries_created") or 0),
        "run_upserted": int(sync_status.get("run_upserted") or 0),
        "run_payload_nodes_upserted": int(sync_status.get("run_payload_nodes_upserted") or 0),
        "documents_total": int(sync_status.get("documents_total") or 0),
        "documents_upserted": int(sync_status.get("documents_upserted") or 0),
        "document_payload_nodes_upserted": int(sync_status.get("document_payload_nodes_upserted") or 0),
        "identifiers_upserted": int(sync_status.get("identifiers_upserted") or 0),
        "classification_scores_upserted": int(sync_status.get("classification_scores_upserted") or 0),
        "classification_score_audit_upserted": int(sync_status.get("classification_score_audit_upserted") or 0),
        "classification_keyword_matches_upserted": int(sync_status.get("classification_keyword_matches_upserted") or 0),
        "anti_confusion_hits_upserted": int(sync_status.get("anti_confusion_hits_upserted") or 0),
        "anti_confusion_targets_upserted": int(sync_status.get("anti_confusion_targets_upserted") or 0),
        "pages_meta_upserted": int(sync_status.get("pages_meta_upserted") or 0),
        "text_normalization_items_upserted": int(sync_status.get("text_normalization_items_upserted") or 0),
        "search_keywords_upserted": int(sync_status.get("search_keywords_upserted") or 0),
        "business_fields_upserted": int(sync_status.get("business_fields_upserted") or 0),
        "sentence_layouts_upserted": int(sync_status.get("sentence_layouts_upserted") or 0),
        "sentence_spans_upserted": int(sync_status.get("sentence_spans_upserted") or 0),
        "layout_header_rows_upserted": int(sync_status.get("layout_header_rows_upserted") or 0),
        "layout_header_cells_upserted": int(sync_status.get("layout_header_cells_upserted") or 0),
        "layout_table_rows_upserted": int(sync_status.get("layout_table_rows_upserted") or 0),
        "human_review_tasks_upserted": int(sync_status.get("human_review_tasks_upserted") or 0),
        "processing_warnings_upserted": int(sync_status.get("processing_warnings_upserted") or 0),
        "processing_logs_upserted": int(sync_status.get("processing_logs_upserted") or 0),
        "processing_steps_upserted": int(sync_status.get("processing_steps_upserted") or 0),
        "processing_durations_upserted": int(sync_status.get("processing_durations_upserted") or 0),
        "component_metrics_upserted": int(sync_status.get("component_metrics_upserted") or 0),
        "links_total": int(sync_status.get("links_total") or 0),
        "links_upserted": int(sync_status.get("links_upserted") or 0),
        "link_shared_terms_upserted": int(sync_status.get("link_shared_terms_upserted") or 0),
        "stable_registry_upserted": int(sync_status.get("stable_registry_upserted") or 0),
        "database_created": bool(sync_status.get("database_created")),
        "schema_ready": bool(sync_status.get("schema_ready")),
        "tables_ready": _safe_list(sync_status.get("tables_ready")),
        "tables_missing": _safe_list(sync_status.get("tables_missing")),
        "config_path": sync_status.get("config_path"),
        "schema_config_path": sync_status.get("schema_config_path"),
        "fusion_path": sync_status.get("fusion_path"),
        "fusion_audit_written": bool(sync_status.get("fusion_audit_written")),
        "skipped": sync_status.get("skipped"),
        "error": sync_status.get("error"),
    }


def attach_postgres_sync_audit(payload: Any, sync_status: Dict[str, Any]) -> Any:
    if not isinstance(payload, dict):
        return payload
    payload["postgres_sync"] = build_postgres_sync_audit(sync_status)
    return payload


def sync_fusion_payload_to_postgres(
    repo_root: Path,
    payload: Any,
    *,
    bootstrap_status: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    pipeline_profile: Optional[str] = None,
    source: Optional[str] = None,
    fusion_path: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg = load_postgres_connection_config(repo_root)
    schema = load_postgres_schema_config(cfg.schema_config_path)
    status: Dict[str, Any] = {
        "enabled": cfg.enabled,
        "sync_enabled": cfg.sync_enabled,
        "sync_write_fusion_audit": cfg.sync_write_fusion_audit,
        "ready": False,
        "host": cfg.host,
        "port": cfg.port,
        "user": cfg.user,
        "database": schema.database_name,
        "config_path": str(cfg.config_path),
        "schema_config_path": str(schema.config_path),
        "run_id": None,
        "pipeline_profile": str(pipeline_profile or "").strip() or None,
        "source": str(source or "").strip() or None,
        "log_entries_created": 0,
        "run_upserted": 0,
        "documents_total": 0,
        "documents_upserted": 0,
        "links_total": 0,
        "links_upserted": 0,
        "ingest_rows_upserted": 0,
        "texts_upserted": 0,
        "payloads_upserted": 0,
        "run_payload_nodes_upserted": 0,
        "document_payload_nodes_upserted": 0,
        "identifiers_upserted": 0,
        "classification_scores_upserted": 0,
        "classification_score_audit_upserted": 0,
        "classification_keyword_matches_upserted": 0,
        "anti_confusion_hits_upserted": 0,
        "anti_confusion_targets_upserted": 0,
        "extractions_upserted": 0,
        "extraction_summaries_upserted": 0,
        "extraction_details_upserted": 0,
        "regex_fields_upserted": 0,
        "regex_matches_upserted": 0,
        "bm25_chunks_upserted": 0,
        "relations_upserted": 0,
        "tables_upserted": 0,
        "table_rows_upserted": 0,
        "table_cells_upserted": 0,
        "quality_checks_upserted": 0,
        "quality_issue_locations_upserted": 0,
        "quality_row_audit_upserted": 0,
        "quality_declared_locations_upserted": 0,
        "quality_check_steps_upserted": 0,
        "component_audit_upserted": 0,
        "pages_upserted": 0,
        "pages_meta_upserted": 0,
        "structure_rows_upserted": 0,
        "visual_marks_upserted": 0,
        "text_normalization_items_upserted": 0,
        "search_keywords_upserted": 0,
        "business_fields_upserted": 0,
        "topics_upserted": 0,
        "sentences_upserted": 0,
        "sentence_layouts_upserted": 0,
        "sentence_spans_upserted": 0,
        "tokens_upserted": 0,
        "entities_upserted": 0,
        "nlp_matches_upserted": 0,
        "layout_header_rows_upserted": 0,
        "layout_header_cells_upserted": 0,
        "layout_table_rows_upserted": 0,
        "vectors_upserted": 0,
        "chunk_embeddings_upserted": 0,
        "word_embeddings_upserted": 0,
        "features_upserted": 0,
        "human_review_tasks_upserted": 0,
        "processing_warnings_upserted": 0,
        "processing_logs_upserted": 0,
        "processing_steps_upserted": 0,
        "processing_durations_upserted": 0,
        "component_metrics_upserted": 0,
        "link_shared_terms_upserted": 0,
        "stable_registry_upserted": 0,
        "link_shared_topics_upserted": 0,
        "link_sentence_matches_upserted": 0,
        "link_chunk_matches_upserted": 0,
        "database_created": False,
        "schema_ready": False,
        "tables_ready": [],
        "tables_missing": [],
        "fusion_path": str(fusion_path) if fusion_path else None,
        "fusion_audit_written": False,
        "skipped": None,
        "error": None,
    }

    if not cfg.enabled:
        status["skipped"] = "disabled"
        return status
    if not cfg.sync_enabled:
        status["skipped"] = "sync_disabled"
        return status
    if not isinstance(payload, dict):
        status["skipped"] = "payload_missing"
        status["error"] = "FUSION_PAYLOAD absent ou invalide"
        return _finalize_sync_or_raise(status, cfg.sync_strict)

    documents = _payload_documents(payload)
    links = _extract_links(payload)
    status["documents_total"] = len(documents)
    status["links_total"] = len(links)
    profile = _normalize_pipeline_profile(
        pipeline_profile
        or payload.get("pipeline_profile")
        or _safe_dict(payload.get("pipeline")).get("profile")
        or payload.get("profile")
        or "pipeline0ml"
    )
    source_value = str(source or payload.get("source") or "fusion-resultats").strip() or "fusion-resultats"
    current_run_id = str(run_id or payload.get("run_id") or "").strip()
    if not current_run_id:
        current_run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    status["run_id"] = current_run_id
    status["pipeline_profile"] = profile
    status["source"] = source_value

    bootstrap = bootstrap_status if isinstance(bootstrap_status, dict) else ensure_postgres_bootstrap(repo_root)
    status["ready"] = bool(bootstrap.get("ready"))
    status["database_created"] = bool(bootstrap.get("database_created"))
    status["schema_ready"] = bool(bootstrap.get("schema_ready"))
    status["tables_ready"] = _safe_list(bootstrap.get("tables_ready"))
    status["tables_missing"] = _safe_list(bootstrap.get("tables_missing"))
    if bootstrap.get("error"):
        status["error"] = str(bootstrap.get("error"))
    if not bootstrap.get("ready"):
        status["skipped"] = "bootstrap_not_ready"
        return _finalize_sync_or_raise(status, cfg.sync_strict)

    now_iso = _utc_now_iso()
    statements: List[str] = ["BEGIN;"]

    if cfg.sync_upsert_runs:
        run_row = _build_run_row(
            payload,
            run_id=current_run_id,
            profile=profile,
            source_value=source_value,
            fusion_path=fusion_path,
            now_iso=now_iso,
        )
        statements.append(
            _build_upsert_statement(
                "dms.runs",
                run_row,
                conflict_columns=["run_id"],
                json_columns=[
                    "raw_payload",
                    "pipeline_json",
                    "postgres_sync_json",
                    "null_policy_json",
                    "source_context_json",
                    "cross_document_analysis_json",
                    "registries_json",
                    "item_templates_json",
                    "sql_mapping_hints_json",
                ],
                skip_update_columns=["created_at"],
            )
        )
        status["run_upserted"] = 1
        status["run_payload_nodes_upserted"] += _append_upsert_rows(
            statements,
            "dms.run_payload_nodes",
            _build_run_payload_node_rows(payload, run_id=current_run_id, now_iso=now_iso),
            conflict_columns=["node_id"],
            json_columns=["value_json", "payload_json"],
            skip_update_columns=["created_at"],
        )

    for index, doc in enumerate(documents, start=1):
        document_id = _extract_document_id(doc, index)
        file_sha256 = _sha256_file(_extract_primary_path(doc))

        if cfg.sync_upsert_documents:
            document_row = _build_document_row(
                doc,
                run_id=current_run_id,
                profile=profile,
                source_value=source_value,
                document_id=document_id,
                now_iso=now_iso,
            )
            status["documents_upserted"] += _append_upsert_rows(
                statements,
                "dms.documents",
                [document_row],
                conflict_columns=["document_id"],
                json_columns=[
                    "file_paths_json",
                    "languages_json",
                    "warnings_json",
                    "logs_json",
                    "visual_flags_json",
                    "classification_scores_json",
                    "classification_keyword_matches_json",
                    "classification_log_json",
                    "classification_scores_audit_json",
                    "anti_confusion_targets_json",
                    "classification_decision_debug_json",
                ],
                skip_update_columns=["created_at"],
            )

            text_row = _build_document_text_row(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso)
            status["texts_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_texts",
                [text_row],
                conflict_columns=["document_id"],
                json_columns=["search_keywords_json", "normalization_json", "text_json"],
                skip_update_columns=["created_at"],
            )

            payload_row = _build_document_payload_row(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso)
            payload_row["payload_schema_version"] = str(payload.get("schema_version") or "").strip() or None
            status["payloads_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_payloads",
                [payload_row],
                conflict_columns=["document_id"],
                json_columns=[
                    "raw_document_json",
                    "source_payload_json",
                    "file_json",
                    "classification_json",
                    "content_json",
                    "text_json",
                    "document_structure_json",
                    "extraction_json",
                    "nlp_json",
                    "ml50_json",
                    "ml100_json",
                    "components_json",
                    "quality_checks_json",
                    "cross_document_json",
                    "processing_json",
                    "meta_json",
                    "ocr_json",
                    "human_review_json",
                ],
                skip_update_columns=["created_at"],
            )
            status["document_payload_nodes_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_payload_nodes",
                _build_document_payload_node_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso),
                conflict_columns=["node_id"],
                json_columns=["value_json", "payload_json"],
                skip_update_columns=["created_at"],
            )

            identifier_rows = _build_document_identifier_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso)
            status["identifiers_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_identifiers",
                identifier_rows,
                conflict_columns=["identifier_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )

            cls_score_rows, cls_audit_rows, cls_keyword_rows, anti_hit_rows, anti_target_rows = _build_document_classification_rows(
                doc,
                run_id=current_run_id,
                document_id=document_id,
                now_iso=now_iso,
            )
            status["classification_scores_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_classification_scores",
                cls_score_rows,
                conflict_columns=["classification_score_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["classification_score_audit_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_classification_score_audit",
                cls_audit_rows,
                conflict_columns=["score_audit_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["classification_keyword_matches_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_classification_keyword_matches",
                cls_keyword_rows,
                conflict_columns=["keyword_match_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["anti_confusion_hits_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_anti_confusion_hits",
                anti_hit_rows,
                conflict_columns=["anti_confusion_hit_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["anti_confusion_targets_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_anti_confusion_targets",
                anti_target_rows,
                conflict_columns=["anti_confusion_target_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )

            extraction_row = _build_document_extraction_summary_row(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso)
            status["extractions_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_extraction_summaries",
                [extraction_row],
                conflict_columns=["document_id"],
                json_columns=[
                    "native_json",
                    "tesseract_json",
                    "regex_extractions_json",
                    "business_json",
                    "relations_json",
                    "bm25_json",
                    "table_extraction_json",
                    "totals_verification_json",
                    "visual_detection_json",
                    "extraction_json",
                ],
                skip_update_columns=["created_at"],
            )
            status["extraction_summaries_upserted"] = int(status.get("extractions_upserted") or 0)

            ingest_row = _build_ingest_row(doc, run_id=current_run_id, document_id=document_id, file_sha256=file_sha256, now_iso=now_iso)
            status["ingest_rows_upserted"] += _append_upsert_rows(
                statements,
                "dms.ingest_queue",
                [ingest_row],
                conflict_columns=["ingest_key"],
                json_columns=["payload_json"],
                skip_update_columns=["received_at"],
            )

            status["component_audit_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_component_audit",
                _build_document_component_audit_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso),
                conflict_columns=["component_audit_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )

            status["pages_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_pages",
                _build_document_page_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso),
                conflict_columns=["page_id"],
                json_columns=["raw_page_json"],
                skip_update_columns=["created_at"],
            )
            status["pages_meta_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_pages_meta",
                _build_document_page_meta_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso),
                conflict_columns=["page_meta_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )

            for table_name, rows in _build_document_structure_detail_row_sets(
                doc,
                run_id=current_run_id,
                document_id=document_id,
                now_iso=now_iso,
            ).items():
                status["structure_rows_upserted"] += _append_upsert_rows(
                    statements,
                    table_name,
                    rows,
                    conflict_columns=[next(iter(rows[0].keys()))] if rows else ["dummy_id"],
                    json_columns=["raw_json"],
                    skip_update_columns=["created_at"],
                )

            status["visual_marks_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_visual_marks",
                _build_document_visual_mark_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso),
                conflict_columns=["visual_mark_id"],
                json_columns=["bbox_px_json", "bbox_norm_json", "payload_json"],
                skip_update_columns=["created_at"],
            )

            normalization_rows, search_keyword_rows = _build_document_text_aux_rows(
                doc,
                run_id=current_run_id,
                document_id=document_id,
                now_iso=now_iso,
            )
            status["text_normalization_items_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_text_normalization_items",
                normalization_rows,
                conflict_columns=["normalization_item_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["search_keywords_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_search_keywords",
                search_keyword_rows,
                conflict_columns=["search_keyword_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["business_fields_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_business_fields",
                _build_document_business_field_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso),
                conflict_columns=["business_field_id"],
                json_columns=["value_json", "payload_json"],
                skip_update_columns=["created_at"],
            )

            table_rows, row_rows = _build_document_table_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso)
            status["tables_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_tables",
                table_rows,
                conflict_columns=["document_table_id"],
                json_columns=["header_map_json", "detected_columns_json", "totals_json", "shape_json", "raw_table_json"],
                skip_update_columns=["created_at"],
            )
            status["table_rows_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_table_rows",
                row_rows,
                conflict_columns=["table_row_id"],
                json_columns=["raw_cells_json", "raw_row_json"],
                skip_update_columns=["created_at"],
            )
            status["table_cells_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_table_cells",
                _build_document_table_cell_rows(
                    row_rows,
                    run_id=current_run_id,
                    document_id=document_id,
                    now_iso=now_iso,
                ),
                conflict_columns=["table_cell_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )

            quality_rows, quality_issue_rows, quality_row_audit_rows, quality_declared_rows, quality_step_rows = _build_document_quality_rows(
                doc,
                run_id=current_run_id,
                document_id=document_id,
                now_iso=now_iso,
            )
            status["quality_checks_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_quality_checks",
                quality_rows,
                conflict_columns=["quality_check_id"],
                json_columns=[
                    "declared_totals_raw_json",
                    "table_anchor_json",
                    "subtotal_location_json",
                    "tax_location_json",
                    "total_location_json",
                    "issue_locations_json",
                    "details_json",
                    "raw_check_json",
                ],
                skip_update_columns=["created_at"],
            )
            status["quality_issue_locations_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_quality_issue_locations",
                quality_issue_rows,
                conflict_columns=["issue_id"],
                json_columns=["source_location_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            status["quality_row_audit_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_quality_row_audit",
                quality_row_audit_rows,
                conflict_columns=["row_audit_id"],
                json_columns=["source_location_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            status["quality_declared_locations_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_quality_declared_locations",
                quality_declared_rows,
                conflict_columns=["declared_location_id"],
                json_columns=["source_location_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            status["quality_check_steps_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_quality_check_steps",
                quality_step_rows,
                conflict_columns=["check_step_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )

            detail_rows, regex_field_rows, regex_match_rows, bm25_rows, relation_rows = _build_document_extraction_detail_rows(
                doc,
                run_id=current_run_id,
                document_id=document_id,
                now_iso=now_iso,
            )
            status["extraction_details_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_extractions",
                detail_rows,
                conflict_columns=["extraction_id"],
                json_columns=["value_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            status["regex_fields_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_regex_fields",
                regex_field_rows,
                conflict_columns=["regex_field_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["regex_matches_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_regex_matches",
                regex_match_rows,
                conflict_columns=["regex_match_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["bm25_chunks_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_bm25_chunks",
                bm25_rows,
                conflict_columns=["bm25_chunk_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["relations_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_relations",
                relation_rows,
                conflict_columns=["relation_id"],
                json_columns=["source_location_json", "payload_json"],
                skip_update_columns=["created_at"],
            )

            topic_rows = _build_document_topic_rows(
                doc,
                run_id=current_run_id,
                document_id=document_id,
                profile=profile,
                now_iso=now_iso,
            )
            status["topics_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_topics",
                topic_rows,
                conflict_columns=["topic_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            sentence_rows = _build_document_sentence_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso)
            status["sentences_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_sentences",
                sentence_rows,
                conflict_columns=["sentence_id"],
                json_columns=["source_location_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            sentence_layout_rows, sentence_span_rows, layout_header_rows, layout_header_cells, layout_table_rows = _build_document_sentence_layout_row_sets(
                doc,
                run_id=current_run_id,
                document_id=document_id,
                now_iso=now_iso,
                valid_sentence_ids={str(row.get("sentence_id")) for row in sentence_rows if row.get("sentence_id")},
            )
            status["sentence_layouts_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_sentence_layouts",
                sentence_layout_rows,
                conflict_columns=["sentence_layout_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["sentence_spans_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_sentence_spans",
                sentence_span_rows,
                conflict_columns=["sentence_span_id"],
                json_columns=["bbox_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            status["layout_header_rows_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_layout_header_rows",
                layout_header_rows,
                conflict_columns=["header_row_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["layout_header_cells_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_layout_header_cells",
                layout_header_cells,
                conflict_columns=["header_cell_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["layout_table_rows_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_layout_table_rows",
                layout_table_rows,
                conflict_columns=["layout_table_row_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            entity_rows = _build_document_entity_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso)
            status["entities_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_entities",
                entity_rows,
                conflict_columns=["entity_id"],
                json_columns=["source_location_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            token_rows = _build_document_token_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso)
            status["tokens_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_tokens",
                token_rows,
                conflict_columns=["token_id"],
                json_columns=["source_location_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            nlp_match_rows = _build_document_nlp_match_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso)
            status["nlp_matches_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_nlp_matches",
                nlp_match_rows,
                conflict_columns=["nlp_match_id"],
                json_columns=[
                    "source_location_json",
                    "shared_terms_json",
                    "shared_topics_json",
                    "phrase_a_json",
                    "phrase_b_json",
                    "chunk_a_json",
                    "chunk_b_json",
                    "payload_json",
                ],
                skip_update_columns=["created_at"],
            )
            status["vectors_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_vectors",
                _build_document_vector_rows(
                    doc,
                    run_id=current_run_id,
                    document_id=document_id,
                    profile=profile,
                    now_iso=now_iso,
                ),
                conflict_columns=["vector_id"],
                json_columns=["vector_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            status["chunk_embeddings_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_chunk_embeddings",
                _build_document_chunk_embedding_rows(
                    doc,
                    run_id=current_run_id,
                    document_id=document_id,
                    profile=profile,
                    now_iso=now_iso,
                ),
                conflict_columns=["chunk_embedding_id"],
                json_columns=["chunk_topics_json", "vector_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            status["word_embeddings_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_word_embeddings",
                _build_document_word_embedding_rows(
                    doc,
                    run_id=current_run_id,
                    document_id=document_id,
                    profile=profile,
                    now_iso=now_iso,
                ),
                conflict_columns=["word_embedding_id"],
                json_columns=["vector_json", "payload_json"],
                skip_update_columns=["created_at"],
            )

            feature_rows = _build_document_feature_rows(
                doc,
                run_id=current_run_id,
                document_id=document_id,
                profile=profile,
                now_iso=now_iso,
            )
            status["features_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_pipeline_features",
                feature_rows,
                conflict_columns=["feature_id"],
                json_columns=["topics_json", "payload_json"],
                skip_update_columns=["created_at"],
            )
            status["human_review_tasks_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_human_review_tasks",
                _build_document_human_review_task_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso),
                conflict_columns=["human_review_task_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            processing_warning_rows, processing_log_rows, processing_step_rows, processing_duration_rows = _build_document_processing_rows(
                doc,
                run_id=current_run_id,
                document_id=document_id,
                now_iso=now_iso,
            )
            status["processing_warnings_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_processing_warnings",
                processing_warning_rows,
                conflict_columns=["processing_warning_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["processing_logs_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_processing_logs",
                processing_log_rows,
                conflict_columns=["processing_log_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["processing_steps_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_processing_steps",
                processing_step_rows,
                conflict_columns=["processing_step_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["processing_durations_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_processing_durations",
                processing_duration_rows,
                conflict_columns=["processing_duration_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at"],
            )
            status["component_metrics_upserted"] += _append_upsert_rows(
                statements,
                "dms.document_component_metrics",
                _build_document_component_metric_rows(doc, run_id=current_run_id, document_id=document_id, now_iso=now_iso),
                conflict_columns=["component_metric_id"],
                json_columns=["metric_value_json", "payload_json"],
                skip_update_columns=["created_at"],
            )

            status["stable_registry_upserted"] += _append_upsert_rows(
                statements,
                "dms.stable_id_registry",
                _build_stable_registry_rows(
                    run_id=current_run_id,
                    now_iso=now_iso,
                    document_row=document_row,
                    identifier_rows=identifier_rows,
                    entity_rows=entity_rows,
                    topic_rows=topic_rows,
                    table_rows=table_rows,
                    relation_rows=relation_rows,
                ),
                conflict_columns=["registry_id"],
                json_columns=["payload_json"],
                skip_update_columns=["created_at", "first_seen_run_id"],
            )

    if cfg.sync_upsert_links:
        link_rows, link_topic_rows, link_sentence_rows, link_chunk_rows = _build_document_link_rows(payload, run_id=current_run_id, now_iso=now_iso)
        link_shared_term_rows = _build_document_link_shared_term_rows(payload, run_id=current_run_id, now_iso=now_iso)
        status["links_upserted"] += _append_upsert_rows(
            statements,
            "dms.document_links",
            link_rows,
            conflict_columns=["link_id"],
            json_columns=[
                "shared_topics_json",
                "shared_terms_json",
                "score_breakdown_json",
                "audit_json",
                "vector_audit_json",
                "raw_link_json",
            ],
            skip_update_columns=["created_at"],
        )
        status["link_shared_terms_upserted"] += _append_upsert_rows(
            statements,
            "dms.document_link_shared_terms",
            link_shared_term_rows,
            conflict_columns=["link_shared_term_id"],
            json_columns=["doc_a_examples_json", "doc_b_examples_json", "payload_json"],
            skip_update_columns=["created_at"],
        )
        status["link_shared_topics_upserted"] += _append_upsert_rows(
            statements,
            "dms.document_link_shared_topics",
            link_topic_rows,
            conflict_columns=["link_shared_topic_id"],
            json_columns=["doc_a_examples_json", "doc_b_examples_json", "payload_json"],
            skip_update_columns=["created_at"],
        )
        status["link_sentence_matches_upserted"] += _append_upsert_rows(
            statements,
            "dms.document_link_sentence_matches",
            link_sentence_rows,
            conflict_columns=["link_sentence_match_id"],
            json_columns=["shared_terms_json", "shared_topics_json", "phrase_a_json", "phrase_b_json", "payload_json"],
            skip_update_columns=["created_at"],
        )
        status["link_chunk_matches_upserted"] += _append_upsert_rows(
            statements,
            "dms.document_link_chunk_matches",
            link_chunk_rows,
            conflict_columns=["link_chunk_match_id"],
            json_columns=["shared_terms_json", "shared_topics_json", "chunk_a_json", "chunk_b_json", "payload_json"],
            skip_update_columns=["created_at"],
        )
        status["stable_registry_upserted"] += _append_upsert_rows(
            statements,
            "dms.stable_id_registry",
            _build_stable_registry_rows(
                run_id=current_run_id,
                now_iso=now_iso,
                link_rows=link_rows,
            ),
            conflict_columns=["registry_id"],
            json_columns=["payload_json"],
            skip_update_columns=["created_at", "first_seen_run_id"],
        )

    status["log_entries_created"] = int(status.get("ingest_rows_upserted") or 0)
    statements.append("COMMIT;")

    try:
        _run_exec_sql_stdin(cfg, schema.database_name, "\n\n".join(statements) + "\n")
    except Exception as exc:
        status["error"] = str(exc)
        return _finalize_sync_or_raise(status, cfg.sync_strict)

    return _finalize_sync_or_raise(status, cfg.sync_strict)
