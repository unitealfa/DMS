from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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
_BOOTSTRAP_ATTEMPTED: set[str] = set()


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


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


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
        return repr(float(value))
    except Exception:
        return "NULL"


def _sql_nullable_timestamp(value: Any) -> str:
    if value is None:
        return "NULL"
    text = str(value).strip()
    if not text:
        return "NULL"
    return _quote_sql_literal(text)


def _stable_hash_id(prefix: str, *parts: Any) -> str:
    seed = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"{prefix}-{digest}"


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
    structure = _safe_dict(doc.get("document_structure"))
    pages = _safe_list(structure.get("pages"))
    if pages:
        return len(pages)
    return None


def _extract_links(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    analysis = _safe_dict(payload.get("cross_document_analysis"))
    return [row for row in _safe_list(analysis.get("links")) if isinstance(row, dict)]


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


def _quote_sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quote_sql_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


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
    ok, _, detail = _run_command(cmd, timeout_seconds=max(cfg.connect_timeout, 120), env=_pg_env(cfg, database), stdin_text=sql)
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


def _table_exists(cfg: PostgresConnectionConfig, database_name: str, table_name: str) -> bool:
    sql = (
        "SELECT 1 FROM information_schema.tables "
        f"WHERE table_schema = 'public' AND table_name = {_quote_sql_literal(table_name)};"
    )
    return _run_scalar_sql(cfg, database_name, sql) == "1"


def _apply_schema(cfg: PostgresConnectionConfig, schema: PostgresSchemaConfig, status: Dict[str, Any]) -> None:
    applied_tables: List[str] = []
    for ext in schema.extensions:
        ext_name = str(ext).strip()
        if not ext_name:
            continue
        _run_exec_sql(cfg, schema.database_name, f"CREATE EXTENSION IF NOT EXISTS {_quote_sql_ident(ext_name)};")

    for entry in schema.tables:
        sql = str(entry.get("sql") or "").strip()
        name = str(entry.get("name") or "").strip()
        if not sql or not name:
            continue
        _run_exec_sql(cfg, schema.database_name, sql)
        if _table_exists(cfg, schema.database_name, name):
            applied_tables.append(name)

    for entry in schema.indexes:
        sql = str(entry.get("sql") or "").strip()
        if sql:
            _run_exec_sql(cfg, schema.database_name, sql)

    for sql in schema.post_sql:
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
        "documents_total": int(sync_status.get("documents_total") or 0),
        "documents_upserted": int(sync_status.get("documents_upserted") or 0),
        "links_total": int(sync_status.get("links_total") or 0),
        "links_upserted": int(sync_status.get("links_upserted") or 0),
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
    status["documents_total"] = len(documents)
    status["links_total"] = 0
    profile = str(
        pipeline_profile
        or payload.get("pipeline_profile")
        or _safe_dict(payload.get("pipeline")).get("profile")
        or "default"
    ).strip() or "default"
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

    statements: List[str] = ["BEGIN;"]
    now_iso = _utc_now_iso()
    docs_to_log = documents or [{"file": {"name": "<no-document>"}}]
    for index, doc in enumerate(docs_to_log, start=1):
        filename = _extract_filename(doc) or f"document_{index}"
        log_id = _stable_hash_id("launch", current_run_id, filename, index)
        statements.append(
            f"""
            INSERT INTO dms_pipeline_launch_logs (
                log_id, run_id, pipeline_profile, filename, status_text, launched_at
            ) VALUES (
                {_sql_nullable_text(log_id)},
                {_sql_nullable_text(current_run_id)},
                {_sql_nullable_text(profile)},
                {_sql_nullable_text(filename)},
                'tout est ok',
                {_sql_nullable_timestamp(now_iso)}
            )
            ON CONFLICT (log_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                pipeline_profile = EXCLUDED.pipeline_profile,
                filename = EXCLUDED.filename,
                status_text = EXCLUDED.status_text,
                launched_at = EXCLUDED.launched_at;
            """.strip()
        )
        status["log_entries_created"] += 1

    statements.append("COMMIT;")

    try:
        _run_exec_sql_stdin(cfg, schema.database_name, "\n\n".join(statements) + "\n")
    except Exception as exc:
        status["error"] = str(exc)
        return _finalize_sync_or_raise(status, cfg.sync_strict)

    return _finalize_sync_or_raise(status, cfg.sync_strict)
