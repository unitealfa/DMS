from __future__ import annotations

import argparse
import cgi
import json
import logging
import mimetypes
import os
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote, unquote

from .cli import PIPELINE_DEFAULT_CODE
from .file_resolution import materialize_uploaded_content_from_lfs_pointer
from .orchestrator import create_pipeline_orchestrator, normalize_pipeline_name, pipeline_definition
from .runtime_state import RUNTIME_JOB_ENV, RUNTIME_STATE_ENV, read_runtime_state


REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML_PATH = REPO_ROOT / "index.html"
API_RUNTIME_ROOT = Path(tempfile.gettempdir()) / "dms_api_runtime"
PUBLIC_API_BASE_URL = str(os.environ.get("PUBLIC_API_BASE_URL") or "").strip().rstrip("/")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
API_VERSION = "dms-local-api-2026-04-08-v4"
LOG_PATH_CANDIDATES = [
    REPO_ROOT / "outputgeneralterminal.runtime.txt",
    REPO_ROOT / "outputgeneralterminal.txt",
]
DEFAULT_PIPELINE_ARGS = [
    "--use-elasticsearch",
    "--es-nlp-level",
    "full",
    "--es-nlp-index",
    "dms_nlp_tokens",
]

LOGGER = logging.getLogger(__name__)
JOB_CACHE_LOCK = threading.Lock()
JOB_CACHE: Dict[str, Dict[str, Any]] = {}


def _active_pipeline_profile() -> str:
    raw = os.environ.get("PIPELINE_DEFAULT") or os.environ.get("PIPELINE_PROFILE") or PIPELINE_DEFAULT_CODE
    return normalize_pipeline_name(raw, PIPELINE_DEFAULT_CODE)


def _active_pipeline_source() -> str:
    if os.environ.get("PIPELINE_DEFAULT"):
        return "PIPELINE_DEFAULT"
    if os.environ.get("PIPELINE_PROFILE"):
        return "PIPELINE_PROFILE"
    return "PIPELINE_DEFAULT_CODE"


def _active_pipeline_steps(profile: str | None = None) -> List[str]:
    profile = normalize_pipeline_name(profile, _active_pipeline_profile())
    return create_pipeline_orchestrator(profile, REPO_ROOT, default=PIPELINE_DEFAULT_CODE).list_steps()


def _active_pipeline_metadata(profile: str | None = None) -> Dict[str, Any]:
    profile = normalize_pipeline_name(profile, _active_pipeline_profile())
    definition = pipeline_definition(profile, default=PIPELINE_DEFAULT_CODE)
    orchestrator = create_pipeline_orchestrator(profile, REPO_ROOT, default=PIPELINE_DEFAULT_CODE)
    components = []
    for component in getattr(orchestrator, "components", []):
        script_path = getattr(component, "script", None)
        script_value = ""
        script_name = ""
        if script_path is not None:
            try:
                script_name = Path(script_path).name
                script_value = str(Path(script_path).resolve().relative_to(REPO_ROOT))
            except Exception:
                script_value = str(script_path)
        components.append(
            {
                "step": getattr(component, "name", ""),
                "component_class": component.__class__.__name__,
                "script": script_name,
                "script_path": script_value,
            }
        )

    return {
        "pipeline_profile": profile,
        "pipeline_label": definition.get("label") or profile,
        "pipeline_description": definition.get("description") or "",
        "pipeline_source": _active_pipeline_source(),
        "pipeline_steps": [item["step"] for item in components],
        "pipeline_steps_count": len(components),
        "pipeline_components": components,
        "pipeline_component_count": len(components),
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _job_runtime_dir(job_id: str) -> Path:
    return API_RUNTIME_ROOT / str(job_id).strip()


def _job_inputs_dir(job_id: str) -> Path:
    return _job_runtime_dir(job_id) / "inputs"


def _job_meta_path(job_id: str) -> Path:
    return _job_runtime_dir(job_id) / "meta.json"


def _cache_get(job_id: str) -> Dict[str, Any]:
    with JOB_CACHE_LOCK:
        row = JOB_CACHE.get(str(job_id).strip()) or {}
        return dict(row) if isinstance(row, dict) else {}


def _cache_merge(job_id: str, **updates: Any) -> Dict[str, Any]:
    clean_job_id = str(job_id).strip()
    with JOB_CACHE_LOCK:
        row = JOB_CACHE.get(clean_job_id)
        if not isinstance(row, dict):
            row = {"job_id": clean_job_id}
            JOB_CACHE[clean_job_id] = row
        row.update(updates)
        return dict(row)


def _cleanup_job_runtime(job_id: str) -> None:
    runtime_dir = _job_runtime_dir(job_id)
    if not runtime_dir.exists():
        return
    shutil.rmtree(runtime_dir, ignore_errors=True)


def _compact_manifest_for_cache(manifest: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        return {}
    compact = json.loads(json.dumps(manifest, ensure_ascii=False))
    compact["storage_root"] = None
    compact["runtime_root"] = None
    compact["storage_mode"] = "temporary-no-persistence"
    compact["artifacts_purged"] = True
    for row in compact.get("documents") or []:
        if not isinstance(row, dict):
            continue
        row["temporary_storage"] = True
        row["file_available"] = False
        row["stored_relative_path"] = None
        row["stored_absolute_path"] = None
        row["api_route"] = None
        row["api_url"] = None
        row["download_url"] = None
    return compact


def _mark_job_result_delivered(job_id: str, *, mode: str, reason: str) -> Dict[str, Any]:
    now = _iso_now()
    cached = _cache_get(job_id)
    manifest = _compact_manifest_for_cache(cached.get("manifest"))
    if manifest:
        manifest["result_delivery"] = {
            "delivered": True,
            "mode": mode,
            "reason": reason,
            "delivered_at": now,
        }
    updates = {
        "manifest": manifest or cached.get("manifest") or {},
        "result": None,
        "documents": [],
        "result_delivered": True,
        "result_delivery_mode": mode,
        "result_delivered_at": now,
        "artifacts_purged": True,
    }
    return _cache_merge(job_id, **updates)


def _tail_recent_lines(path: Path, limit: int = 160, offset: int = 0) -> List[str]:
    if not path.exists():
        return []
    try:
        file_size = path.stat().st_size
        safe_offset = offset if 0 <= offset <= file_size else 0
        with path.open("rb") as fh:
            if safe_offset:
                fh.seek(safe_offset)
            raw = fh.read()
        lines = raw.decode("utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-limit:]


def _metadata_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pipeline_profile": snapshot.get("pipeline_profile"),
        "pipeline_label": snapshot.get("pipeline_label"),
        "pipeline_description": snapshot.get("pipeline_description"),
        "pipeline_source": snapshot.get("pipeline_source"),
        "pipeline_steps": list(snapshot.get("pipeline_steps") or []),
        "pipeline_steps_count": int(snapshot.get("pipeline_steps_count") or 0),
        "pipeline_components": list(snapshot.get("pipeline_components") or []),
        "pipeline_component_count": int(snapshot.get("pipeline_component_count") or 0),
    }


def _current_runtime_progress(
    status: str,
    metadata: Dict[str, Any] | None = None,
    log_offsets: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = metadata or _active_pipeline_metadata()
    steps = list(metadata.get("pipeline_steps") or [])
    if status == "idle":
        return {
            **metadata,
            "completed_steps_count": 0,
            "current_step": None,
            "step_index": 0,
            "steps_total": len(steps),
            "progress_state": {
                "step_index": 0,
                "steps_total": len(steps),
            },
            "component_name": None,
            "component_script": None,
            "component_status": None,
            "progress_percent": 0,
            "last_log_line": None,
            "log_path": None,
        }
    log_offsets = log_offsets or {}
    log_path = None
    lines: List[str] = []

    for candidate in LOG_PATH_CANDIDATES:
        if not candidate.exists():
            continue
        if log_path is None:
            log_path = candidate
        offset = int(log_offsets.get(str(candidate), 0) or 0)
        candidate_lines = _tail_recent_lines(candidate, offset=offset)
        if candidate_lines:
            log_path = candidate
            lines = candidate_lines
            break

    current_step = None
    last_log_line = None

    for line in lines:
        if line.strip():
            last_log_line = line.strip()
        marker = "Execution du composant "
        if marker in line:
            current_step = line.split(marker, 1)[1].split(" via", 1)[0].strip()

    progress_percent = 0
    completed_steps = 0
    if status == "completed":
        progress_percent = 100
        completed_steps = len(steps)
        current_step = current_step or (steps[-1] if steps else None)
    elif current_step and current_step in steps:
        step_index = steps.index(current_step)
        completed_steps = step_index
        progress_percent = max(3, min(97, int(((step_index + 0.5) / max(len(steps), 1)) * 100)))

    component_meta = next(
        (item for item in metadata.get("pipeline_components") or [] if item.get("step") == current_step),
        None,
    )
    step_index = steps.index(current_step) + 1 if current_step in steps else (len(steps) if status == "completed" else 0)
    component_script = None
    if component_meta:
        component_script = component_meta.get("script_path") or component_meta.get("script") or None

    component_status = None
    if current_step:
        if status == "running":
            component_status = "running"
        elif status == "completed":
            component_status = "completed"
        elif status == "failed":
            component_status = "failed"

    return {
        **metadata,
        "completed_steps_count": completed_steps,
        "current_step": current_step,
        "step_index": step_index,
        "steps_total": len(steps),
        "progress_state": {
            "step_index": step_index,
            "steps_total": len(steps),
        },
        "component_name": current_step,
        "component_script": component_script,
        "component_status": component_status,
        "progress_percent": progress_percent,
        "last_log_line": last_log_line,
        "log_path": str(log_path) if log_path else None,
    }


def _discover_ipv4_addresses() -> List[str]:
    found: List[str] = []
    seen = set()

    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)
        for info in infos:
            ip = str(info[4][0] or "").strip()
            if not ip or ip.startswith("127.") or ip in seen:
                continue
            seen.add(ip)
            found.append(ip)
    except Exception:
        pass

    return found


def _candidate_urls(host: str, port: int) -> List[str]:
    clean_host = str(host or "").strip() or DEFAULT_HOST
    urls: List[str] = []

    if clean_host in {"0.0.0.0", "::"}:
        urls.append(f"http://127.0.0.1:{port}")
        for ip in _discover_ipv4_addresses():
            urls.append(f"http://{ip}:{port}")
        if not urls:
            urls.append(f"http://localhost:{port}")
        return urls

    return [f"http://{clean_host}:{port}"]


def _sanitize_filename(name: str, index: int) -> str:
    base = Path(str(name or "")).name.strip()
    if not base:
        base = f"upload_{index}"
    return base.replace("\x00", "")


def _extract_uploaded_payload(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_type = handler.headers.get("Content-Type", "")
    LOGGER.info(
        "POST %s content_type=%s content_length=%s",
        getattr(handler, "path", ""),
        handler.headers.get("Content-Type"),
        handler.headers.get("Content-Length"),
    )
    if "multipart/form-data" not in content_type:
        raise ValueError("Content-Type multipart/form-data requis.")

    form = cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": handler.headers.get("Content-Length", "0"),
        },
    )
    LOGGER.info("multipart fields=%s", [getattr(item, "name", None) for item in (form.list or [])])

    raw_items = None
    for field_name in ("files", "files[]", "file"):
        if field_name in form:
            raw_items = form[field_name]
            break
    if raw_items is None:
        fallback_items = []
        for item in form.list or []:
            if getattr(item, "filename", None):
                fallback_items.append(item)
        raw_items = fallback_items

    if raw_items is None or (isinstance(raw_items, list) and not raw_items):
        raise ValueError(
            f"Aucun champ fichier recu. Content-Type={content_type}. "
            f"Fields={[getattr(item, 'name', None) for item in (form.list or [])]}"
        )

    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    items: List[Dict[str, Any]] = []
    for idx, field in enumerate(raw_items, start=1):
        if not getattr(field, "file", None):
            continue
        payload = field.file.read()
        items.append(
            {
                "index": idx,
                "filename": _sanitize_filename(getattr(field, "filename", ""), idx),
                "content": payload,
            }
        )
    LOGGER.info("extracted_files_count=%s", len(items))
    text_fields: Dict[str, Any] = {}
    for item in form.list or []:
        name = str(getattr(item, "name", "") or "").strip()
        if not name or getattr(item, "filename", None):
            continue
        value = str(getattr(item, "value", "") or "").strip()
        if not value:
            continue
        if name in text_fields:
            current = text_fields[name]
            if isinstance(current, list):
                current.append(value)
            else:
                text_fields[name] = [current, value]
        else:
            text_fields[name] = value
    return {
        "files": items,
        "fields": text_fields,
    }


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _request_origin(handler: BaseHTTPRequestHandler) -> str:
    proto = (handler.headers.get("X-Forwarded-Proto") or "http").strip() or "http"
    host = (handler.headers.get("X-Forwarded-Host") or handler.headers.get("Host") or "").strip()
    return f"{proto}://{host}" if host else ""


def _first_text_field(payload: Any) -> str:
    if isinstance(payload, list):
        for item in payload:
            value = str(item or "").strip()
            if value:
                return value
        return ""
    return str(payload or "").strip()


def _public_api_base_url(request_origin: str = "") -> str:
    if PUBLIC_API_BASE_URL:
        return PUBLIC_API_BASE_URL
    return str(request_origin or "").strip().rstrip("/")


def _stored_manifest_path(job_id: str) -> Path:
    return _job_runtime_dir(job_id) / "manifest.json"


def _stored_result_path(job_id: str) -> Path:
    return _job_runtime_dir(job_id) / "result.json"


def _api_file_route(job_id: str, filename: str) -> str:
    return f"/api/documents/file/{quote(job_id)}/{quote(filename)}"


def _api_manifest_route(job_id: str) -> str:
    return f"/api/documents/{quote(job_id)}"


def _api_result_route(job_id: str) -> str:
    return f"/api/result/{quote(job_id)}"


def _build_public_file_url(job_id: str, filename: str, request_origin: str = "") -> str | None:
    base = _public_api_base_url(request_origin)
    if not base:
        return None
    return f"{base}{_api_file_route(job_id, filename)}"


def _build_public_manifest_url(job_id: str, request_origin: str = "") -> str | None:
    base = _public_api_base_url(request_origin)
    if not base:
        return None
    return f"{base}{_api_manifest_route(job_id)}"


def _build_public_result_url(job_id: str, request_origin: str = "") -> str | None:
    base = _public_api_base_url(request_origin)
    if not base:
        return None
    return f"{base}{_api_result_route(job_id)}"


def _load_manifest(job_id: str) -> Dict[str, Any]:
    cached = _cache_get(job_id).get("manifest")
    if isinstance(cached, dict):
        return cached
    path = _stored_manifest_path(job_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            _cache_merge(job_id, manifest=payload)
            return payload
        return {}
    except Exception:
        return {}


def _write_manifest(job_id: str, payload: Dict[str, Any]) -> Path:
    path = _stored_manifest_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _cache_merge(job_id, manifest=payload)
    return path


def _load_result(job_id: str) -> Dict[str, Any]:
    cached = _cache_get(job_id).get("result")
    if isinstance(cached, dict):
        return cached
    path = _stored_result_path(job_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            _cache_merge(job_id, result=payload)
            return payload
        return {}
    except Exception:
        return {}


def _result_info(job_id: str, request_origin: str = "") -> Dict[str, Any]:
    cache_row = _cache_get(job_id)
    result_path = _stored_result_path(job_id)
    manifest = _load_manifest(job_id)
    result_payload = _load_result(job_id)
    result_meta = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    result_url = result_meta.get("url") or _build_public_result_url(job_id, request_origin=request_origin)
    return {
        "result_route": _api_result_route(job_id),
        "result_url": result_url,
        "result_path": str(result_path.resolve()) if result_path.exists() else result_meta.get("path"),
        "result_available": bool(result_payload) or bool(result_path.exists()),
        "result_delivered": bool(cache_row.get("result_delivered")),
        "result_delivery_mode": cache_row.get("result_delivery_mode"),
        "result_delivered_at": cache_row.get("result_delivered_at"),
        "artifacts_purged": bool(cache_row.get("artifacts_purged")),
    }


def save_uploaded_files(
    upload_items: List[Dict[str, Any]],
    job_id: str,
    request_origin: str = "",
    client_ip: str = "",
    callback_url: str = "",
) -> List[Dict[str, Any]]:
    target_dir = _job_inputs_dir(job_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_items: List[Dict[str, Any]] = []
    used_names = set()
    for item in upload_items:
        filename = str(item.get("filename") or "").strip() or f"upload_{item.get('index')}"
        candidate = filename
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        counter = 1
        while candidate.lower() in used_names:
            counter += 1
            candidate = f"{stem}_{counter}{suffix}"
        used_names.add(candidate.lower())
        destination = target_dir / candidate
        content = item.get("content") or b""
        resolved_upload = materialize_uploaded_content_from_lfs_pointer(
            content,
            repo_root=REPO_ROOT,
            preferred_name=candidate,
        )
        if resolved_upload.get("is_lfs_pointer") and not resolved_upload.get("resolved"):
            raise ValueError(
                f"Le fichier uploade {candidate} est un pointeur Git LFS sans binaire local resolvable. "
                "Envoie le vrai fichier, pas le pointeur texte."
            )
        content = resolved_upload.get("content") or content
        with destination.open("wb") as fh:
            fh.write(content)
        mime_type = mimetypes.guess_type(destination.name)[0] or "application/octet-stream"
        relative_path = destination.relative_to(API_RUNTIME_ROOT)
        api_route = _api_file_route(job_id, destination.name)
        api_url = _build_public_file_url(job_id, destination.name, request_origin=request_origin)
        saved_items.append(
            {
                "api_document_id": uuid.uuid4().hex,
                "job_id": job_id,
                "filename": destination.name,
                "absolute_path": str(destination.resolve()),
                "relative_path": str(relative_path),
                "manifest_relative_path": str(_stored_manifest_path(job_id).relative_to(API_RUNTIME_ROOT)),
                "manifest_absolute_path": str(_stored_manifest_path(job_id).resolve()),
                "api_route": api_route,
                "api_url": api_url,
                "download_url": api_url,
                "file_size": destination.stat().st_size,
                "file_ext": destination.suffix.lower() or None,
                "file_mime": mime_type,
                "file_sha256": _sha256_bytes(content),
                "source_kind": "api_upload",
                "source_client": "external_site",
                "source_ip": client_ip or None,
                "resolved_from_lfs_pointer": bool(resolved_upload.get("resolved")),
                "resolved_source_path": resolved_upload.get("resolved_source_path"),
                "stored_path": destination,
            }
        )

    manifest = {
        "job_id": job_id,
        "received_at": _iso_now(),
        "storage_root": None,
        "runtime_root": str(_job_runtime_dir(job_id).resolve()),
        "storage_mode": "temporary-no-persistence",
        "request_origin": request_origin or None,
        "client_ip": client_ip or None,
        "result": {
            "ready": False,
            "generated_at": None,
            "path": str(_stored_result_path(job_id).resolve()),
            "route": _api_result_route(job_id),
            "url": _build_public_result_url(job_id, request_origin=request_origin),
            "documents_count": 0,
            "pipeline_profile": None,
        },
        "callback": {
            "url": callback_url or None,
            "attempted": False,
            "ok": None,
            "status_code": None,
            "error": None,
        },
        "documents": [
            {
                "api_document_id": item["api_document_id"],
                "file_name": item["filename"],
                "file_ext": item["file_ext"],
                "file_mime": item["file_mime"],
                "file_size": item["file_size"],
                "file_sha256": item["file_sha256"],
                "resolved_from_lfs_pointer": item["resolved_from_lfs_pointer"],
                "resolved_source_path": item["resolved_source_path"],
                "stored_relative_path": item["relative_path"],
                "stored_absolute_path": item["absolute_path"],
                "temporary_storage": True,
                "api_route": item["api_route"],
                "api_url": item["api_url"],
                "download_url": item["download_url"],
            }
            for item in saved_items
        ],
    }
    _write_manifest(job_id, manifest)
    _cache_merge(job_id, documents=manifest.get("documents") or [])
    return saved_items


def _documents_index_payload(limit: int = 100) -> Dict[str, Any]:
    with JOB_CACHE_LOCK:
        cached_rows = list(JOB_CACHE.values())
    docs: List[Dict[str, Any]] = []
    for row in reversed(cached_rows[-limit:]):
        if not isinstance(row, dict):
            continue
        manifest = row.get("manifest")
        if isinstance(manifest, dict):
            docs.append(manifest)
    return {
        "storage_root": None,
        "storage_mode": "temporary-no-persistence",
        "jobs_count": len(docs),
        "jobs": docs,
    }


def _send_file_response(handler: BaseHTTPRequestHandler, file_path: Path) -> None:
    LOGGER.info("Resolved file_path=%s", file_path)
    LOGGER.info("exists=%s is_file=%s", file_path.exists(), file_path.is_file())
    if not file_path.exists() or not file_path.is_file():
        LOGGER.warning("Document not found on disk: file_path=%s", file_path)
        handler._send_json({"error": "Document introuvable"}, status=HTTPStatus.NOT_FOUND)
        return
    raw = file_path.read_bytes()
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    handler.send_response(HTTPStatus.OK)
    handler._send_cors_headers()
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Content-Disposition", f'inline; filename="{file_path.name}"')
    handler.end_headers()
    handler.wfile.write(raw)


def build_cli_command(file_paths: List[Path], extra_args: List[str] | None = None) -> List[str]:
    command = [sys.executable, str(REPO_ROOT / "main.py")]
    command.extend(str(path) for path in file_paths)
    command.extend(DEFAULT_PIPELINE_ARGS)
    if extra_args:
        command.extend(extra_args)
    return command


class LauncherState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Dict[str, Any] = {
            "job_id": None,
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "command": [],
            "files": [],
            "pid": None,
            "error": None,
            "upload_dir": None,
            "log_offsets": {},
        }
        self._process: subprocess.Popen[str] | None = None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = dict(self._current)
        base_status = str(snap.get("status") or "idle")
        base_returncode = snap.get("returncode")
        base_finished_at = snap.get("finished_at")
        base_error = snap.get("error")
        log_offsets = dict(snap.pop("log_offsets", {}) or {})
        runtime_state_path = snap.get("runtime_state_path")
        metadata = _metadata_from_snapshot(snap)
        snap["api_version"] = API_VERSION
        runtime_state = read_runtime_state(runtime_state_path)
        if runtime_state:
            runtime_status = str(runtime_state.get("status") or "")
            resolved_status = base_status
            if base_status in {"idle", "running"} and runtime_status in {"completed", "failed"}:
                resolved_status = runtime_status
            snap.update(metadata)
            snap.update(runtime_state)
            snap["status"] = resolved_status or runtime_status or "idle"
            snap["returncode"] = base_returncode
            snap["finished_at"] = base_finished_at or snap.get("finished_at") or runtime_state.get("finished_at")
            snap["error"] = base_error or snap.get("error")
            if snap["status"] == "completed":
                snap["progress_percent"] = 100
                snap["completed_steps_count"] = int(
                    snap.get("completed_steps_count") or metadata.get("pipeline_steps_count") or 0
                )
                if not snap.get("current_step"):
                    steps = list(metadata.get("pipeline_steps") or [])
                    snap["current_step"] = steps[-1] if steps else None
            elif snap["status"] == "idle":
                snap["progress_percent"] = 0
            snap["progress_state"] = {
                "step_index": int(snap.get("step_index") or 0),
                "steps_total": int(snap.get("steps_total") or metadata.get("pipeline_steps_count") or 0),
            }
        else:
            snap.update(_current_runtime_progress(str(snap.get("status") or ""), metadata=metadata, log_offsets=log_offsets))
        job_id = str(snap.get("job_id") or "").strip()
        if job_id:
            snap.update(_result_info(job_id))
        return snap

    def start_job(
        self,
        file_paths: List[Path],
        extra_args: List[str] | None = None,
        job_id: str | None = None,
        pipeline_profile: str | None = None,
        request_origin: str = "",
        callback_url: str = "",
        callback_token: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Un pipeline est deja en cours.")

            job_id = job_id or uuid.uuid4().hex
            command = build_cli_command(file_paths, extra_args=extra_args)
            requested_profile = normalize_pipeline_name(pipeline_profile, _active_pipeline_profile())
            metadata = _active_pipeline_metadata(requested_profile)
            runtime_state_path = Path(tempfile.gettempdir()) / "dms_launcher_runtime" / f"{job_id}.json"
            print(f"[local-api] lancement job={job_id}")
            print(f"[local-api] commande: {' '.join(command)}")
            LOGGER.info("launching job=%s files=%s", job_id, [str(p) for p in file_paths])

            env = os.environ.copy()
            env[RUNTIME_STATE_ENV] = str(runtime_state_path)
            env[RUNTIME_JOB_ENV] = job_id
            env["DMS_API_JOB_ID"] = job_id
            env["DMS_API_MANIFEST_PATH"] = str(_stored_manifest_path(job_id).resolve())
            env["DMS_API_RESULT_PATH"] = str(_stored_result_path(job_id).resolve())
            env["DMS_API_RESULT_ROUTE"] = _api_result_route(job_id)
            env["DMS_API_RESULT_URL"] = _build_public_result_url(job_id, request_origin=request_origin) or ""
            env["DMS_API_CALLBACK_URL"] = callback_url or ""
            env["DMS_API_CALLBACK_TOKEN"] = callback_token or ""
            env["PIPELINE_PROFILE"] = requested_profile
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
            )
            self._process = process
            self._current = {
                "job_id": job_id,
                "status": "running",
                "started_at": _iso_now(),
                "finished_at": None,
                "returncode": None,
                "command": command,
                "files": [str(path) for path in file_paths],
                "pid": process.pid,
                "error": None,
                "upload_dir": str(file_paths[0].parent) if file_paths else None,
                "runtime_state_path": str(runtime_state_path),
                "callback_url": callback_url or None,
                "manifest_route": _api_manifest_route(job_id),
                "manifest_url": _build_public_manifest_url(job_id, request_origin=request_origin),
                "storage_root": None,
                "storage_mode": "temporary-no-persistence",
                "pipeline_profile": requested_profile,
                "log_offsets": {
                    str(path): path.stat().st_size if path.exists() else 0
                    for path in LOG_PATH_CANDIDATES
                },
            }
            self._current.update(metadata)
            self._current.update(_result_info(job_id, request_origin=request_origin))

            watcher = threading.Thread(target=self._wait_for_process, args=(process, job_id), daemon=True)
            watcher.start()
            return dict(self._current)

    def _wait_for_process(self, process: subprocess.Popen[str], job_id: str) -> None:
        returncode = process.wait()
        manifest = _load_manifest(job_id)
        result = _load_result(job_id)
        callback_meta = manifest.get("callback") if isinstance(manifest.get("callback"), dict) else {}
        callback_ok = bool(callback_meta.get("ok"))
        compact_manifest = _compact_manifest_for_cache(manifest)
        _cache_merge(
            job_id,
            manifest=compact_manifest or None,
            result=result or None,
            artifacts_purged=True,
        )
        _cleanup_job_runtime(job_id)
        if callback_ok and returncode == 0:
            _mark_job_result_delivered(job_id, mode="callback", reason="callback_ok")
        with self._lock:
            if self._current.get("job_id") != job_id:
                return
            self._current["returncode"] = returncode
            self._current["finished_at"] = _iso_now()
            self._current["status"] = "completed" if returncode == 0 else "failed"
            self._current["pid"] = None
            self._process = None
        print(f"[local-api] job={job_id} termine rc={returncode}")


class DMSLauncherHandler(BaseHTTPRequestHandler):
    server_version = "DMSLauncher/1.0"

    @property
    def launcher_state(self) -> LauncherState:
        return self.server.launcher_state  # type: ignore[attr-defined]

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        raw = _json_bytes(payload)
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html_file(self, path: Path) -> None:
        raw = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[local-api] " + (fmt % args) + "\n")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_html_file(INDEX_HTML_PATH)
            return

        if self.path == "/api/status":
            self._send_json(self.launcher_state.snapshot())
            return

        if self.path.startswith("/api/result/"):
            job_id = unquote(self.path[len("/api/result/"):].strip("/"))
            if not job_id:
                self._send_json({"error": "job_id manquant"}, status=HTTPStatus.BAD_REQUEST)
                return
            result = _load_result(job_id)
            if not result:
                self._send_json({"error": "Resultat introuvable"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(result)
            _mark_job_result_delivered(job_id, mode="pull", reason="api_result_read")
            return

        if self.path == "/api/documents":
            self._send_json(_documents_index_payload())
            return

        if self.path.startswith("/api/documents/file/"):
            suffix = self.path[len("/api/documents/file/"):]
            parts = [unquote(part) for part in suffix.split("/") if part]
            if len(parts) < 2:
                self._send_json({"error": "Route document invalide"}, status=HTTPStatus.BAD_REQUEST)
                return
            job_id = parts[0]
            filename = parts[-1]
            job_root = _job_inputs_dir(job_id).resolve()
            file_path = (job_root / filename).resolve()
            LOGGER.info("Serving file: job_id=%s filename=%s", job_id, filename)
            LOGGER.info("Resolved job_root=%s", job_root)
            LOGGER.info("Resolved file_path=%s", file_path)
            LOGGER.info("exists=%s is_file=%s", file_path.exists(), file_path.is_file())
            try:
                file_path.relative_to(job_root)
            except Exception:
                self._send_json({"error": "Chemin document invalide"}, status=HTTPStatus.BAD_REQUEST)
                return
            _send_file_response(self, file_path)
            return

        if self.path.startswith("/api/documents/"):
            job_id = unquote(self.path[len("/api/documents/"):].strip("/"))
            if not job_id:
                self._send_json({"error": "job_id manquant"}, status=HTTPStatus.BAD_REQUEST)
                return
            manifest = _load_manifest(job_id)
            if not manifest:
                self._send_json({"error": "Job introuvable"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(manifest)
            return

        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/run", "/api/store"}:
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        if self.path == "/api/store":
            self._send_json(
                {
                    "ok": False,
                    "error": "POST /api/store est desactive. Les documents ne sont plus stockes de facon persistante.",
                },
                status=HTTPStatus.GONE,
            )
            return

        try:
            request_payload = _extract_uploaded_payload(self)
            upload_items = request_payload.get("files") or []
            request_fields = request_payload.get("fields") or {}
            if not upload_items:
                raise ValueError("Aucun fichier exploitable recu.")
            job_id = uuid.uuid4().hex
            request_origin = _request_origin(self)
            client_ip = str(self.client_address[0] if self.client_address else "").strip()
            requested_pipeline = normalize_pipeline_name(
                _first_text_field(request_fields.get("pipeline")) or self.headers.get("X-Pipeline-Profile") or None,
                _active_pipeline_profile(),
            )
            callback_url = _first_text_field(request_fields.get("callback_url")) or str(
                self.headers.get("X-Callback-Url") or ""
            ).strip()
            callback_token = _first_text_field(request_fields.get("callback_token")) or str(
                self.headers.get("X-Callback-Token") or ""
            ).strip()
            saved_items = save_uploaded_files(
                upload_items,
                job_id,
                request_origin=request_origin,
                client_ip=client_ip,
                callback_url=callback_url,
            )
            LOGGER.info("saved_items=%s", [item["absolute_path"] for item in saved_items])
            stored_documents = [
                {
                    "api_document_id": item["api_document_id"],
                    "job_id": item["job_id"],
                    "file_name": item["filename"],
                    "file_ext": item["file_ext"],
                    "file_mime": item["file_mime"],
                    "content_type": item["file_mime"],
                    "file_size": item["file_size"],
                    "file_sha256": item["file_sha256"],
                    "stored_relative_path": item["relative_path"],
                    "stored_absolute_path": item["absolute_path"],
                    "api_route": item["api_route"],
                    "api_url": item["api_url"],
                    "download_url": item["download_url"],
                }
                for item in saved_items
            ]
            manifest_route = _api_manifest_route(job_id)
            manifest_url = _build_public_manifest_url(job_id, request_origin=request_origin)
            result_route = _api_result_route(job_id)
            result_url = _build_public_result_url(job_id, request_origin=request_origin)

            saved_paths = [Path(item["absolute_path"]) for item in saved_items]
            extra_args = ["--pipeline", requested_pipeline] if requested_pipeline else None
            current = self.launcher_state.start_job(
                saved_paths,
                extra_args=extra_args,
                job_id=job_id,
                pipeline_profile=requested_pipeline,
                request_origin=request_origin,
                callback_url=callback_url,
                callback_token=callback_token,
            )
            current["stored_documents"] = stored_documents
            current["manifest_route"] = manifest_route
            current["manifest_url"] = manifest_url
            current["result_route"] = result_route
            current["result_url"] = result_url
            current["callback_url"] = callback_url or None
            current["storage_root"] = None
            current["storage_mode"] = "temporary-no-persistence"
            self._send_json(
                {
                    "ok": True,
                    "message": "Pipeline lance.",
                    "job": current,
                },
                status=HTTPStatus.ACCEPTED,
            )
        except ValueError as exc:
            LOGGER.warning("POST %s bad request: %s", self.path, exc)
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
        except Exception as exc:
            LOGGER.exception("POST %s failed", self.path)
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


class DMSLauncherServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler]) -> None:
        super().__init__(server_address, handler_class)
        self.launcher_state = LauncherState()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API locale pour lancer le pipeline DMS depuis index.html.")
    parser.add_argument("--host", default=os.environ.get("DMS_API_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DMS_API_PORT", DEFAULT_PORT)))
    return parser.parse_args(argv)


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    try:
        server = DMSLauncherServer((host, port), DMSLauncherHandler)
    except OSError as exc:
        if getattr(exc, "errno", None) == 98:
            raise RuntimeError(
                f"Le port {port} est deja utilise. Lance sur un autre port, par exemple: "
                f"python local_api.py --host {host} --port 8766"
            ) from exc
        raise
    print(f"[local-api] pid={os.getpid()}")
    print(f"[local-api] api_version={API_VERSION}")
    print(f"[local-api] host bind={host}:{port}")
    print(f"[local-api] storage_mode=temporary-no-persistence")
    print(f"[local-api] runtime_root={API_RUNTIME_ROOT.resolve()}")
    for url in _candidate_urls(host, port):
        print(f"[local-api] url={url}")
    print("[local-api] le bouton Lancer de index.html executera main.py avec les fichiers uploades")

    stop_event = {"done": False}

    def _graceful_stop(signum: int, _frame: Any) -> None:
        if stop_event["done"]:
            return
        stop_event["done"] = True
        print(f"[local-api] signal recu={signum} -> arret du serveur")
        threading.Thread(target=server.shutdown, daemon=True).start()

    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        server.server_close()
        print("[local-api] serveur arrete")


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
