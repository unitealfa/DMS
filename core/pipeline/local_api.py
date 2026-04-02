from __future__ import annotations

import argparse
import cgi
import json
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

from .cli import PIPELINE_DEFAULT_CODE, _normalize_pipeline_name
from .orchestrator import Pipeline0MLOrchestrator, Pipeline50MLOrchestrator, Pipeline100MLOrchestrator
from .postgres import ensure_postgres_bootstrap


REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML_PATH = REPO_ROOT / "index.html"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
API_VERSION = "dms-local-api-2026-03-31-v2"
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

PIPELINE_LABELS = {
    "pipeline0ml": "Pipeline 0ML",
    "pipeline50ml": "Pipeline 50ML",
    "pipeline100ml": "Pipeline 100ML",
}

PIPELINE_DESCRIPTIONS = {
    "pipeline0ml": "Baseline non-ML routing with classic tokenisation, grammar, rule extraction and totals verification.",
    "pipeline50ml": "Hybrid ML retrieval pipeline with 50ML tokenisation and extraction components plus grammar refinement.",
    "pipeline100ml": "Transformer-grade pipeline with 100ML tokenisation, XLM-R grammar and visual marks detection.",
}


def _active_pipeline_profile() -> str:
    raw = os.environ.get("PIPELINE_DEFAULT") or os.environ.get("PIPELINE_PROFILE") or PIPELINE_DEFAULT_CODE
    return _normalize_pipeline_name(raw, "pipeline0ml")


def _active_pipeline_source() -> str:
    if os.environ.get("PIPELINE_DEFAULT"):
        return "PIPELINE_DEFAULT"
    if os.environ.get("PIPELINE_PROFILE"):
        return "PIPELINE_PROFILE"
    return "PIPELINE_DEFAULT_CODE"


def _pipeline_orchestrator(profile: str):
    if profile == "pipeline50ml":
        return Pipeline50MLOrchestrator(REPO_ROOT)
    if profile == "pipeline100ml":
        return Pipeline100MLOrchestrator(REPO_ROOT)
    return Pipeline0MLOrchestrator(REPO_ROOT)


def _active_pipeline_steps() -> List[str]:
    profile = _active_pipeline_profile()
    return _pipeline_orchestrator(profile).list_steps()


def _active_pipeline_metadata() -> Dict[str, Any]:
    profile = _active_pipeline_profile()
    orchestrator = _pipeline_orchestrator(profile)
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
        "pipeline_label": PIPELINE_LABELS.get(profile, profile),
        "pipeline_description": PIPELINE_DESCRIPTIONS.get(profile, ""),
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


def _tail_recent_lines(path: Path, limit: int = 160) -> List[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-limit:]


def _current_runtime_progress(status: str) -> Dict[str, Any]:
    metadata = _active_pipeline_metadata()
    steps = list(metadata.get("pipeline_steps") or [])
    log_path = next((path for path in LOG_PATH_CANDIDATES if path.exists()), None)
    lines = _tail_recent_lines(log_path) if log_path else []
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

    return {
        **metadata,
        "completed_steps_count": completed_steps,
        "current_step": current_step,
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


def _extract_uploaded_files(handler: BaseHTTPRequestHandler) -> List[Dict[str, Any]]:
    content_type = handler.headers.get("Content-Type", "")
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
        raise ValueError("Aucun champ fichier recu. Le backend attend 'files' ou 'files[]'.")

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
    return items


def save_uploaded_files(upload_items: List[Dict[str, Any]], job_id: str) -> List[Path]:
    target_dir = Path(tempfile.gettempdir()) / "dms_launcher_uploads" / job_id
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
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
        with destination.open("wb") as fh:
            fh.write(item.get("content") or b"")
        saved_paths.append(destination)
    return saved_paths


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
        }
        self._process: subprocess.Popen[str] | None = None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = dict(self._current)
        snap["api_version"] = API_VERSION
        snap.update(_current_runtime_progress(str(snap.get("status") or "")))
        return snap

    def start_job(
        self,
        file_paths: List[Path],
        extra_args: List[str] | None = None,
        job_id: str | None = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Un pipeline est deja en cours.")

            job_id = job_id or uuid.uuid4().hex
            command = build_cli_command(file_paths, extra_args=extra_args)
            print(f"[local-api] lancement job={job_id}")
            print(f"[local-api] commande: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
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
            }
            self._current.update(_active_pipeline_metadata())

            watcher = threading.Thread(target=self._wait_for_process, args=(process, job_id), daemon=True)
            watcher.start()
            return dict(self._current)

    def _wait_for_process(self, process: subprocess.Popen[str], job_id: str) -> None:
        returncode = process.wait()
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
        if self.path != "/api/run":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            upload_items = _extract_uploaded_files(self)
            if not upload_items:
                raise ValueError("Aucun fichier exploitable recu.")
            job_id = uuid.uuid4().hex
            saved_paths = save_uploaded_files(upload_items, job_id)
            current = self.launcher_state.start_job(saved_paths, job_id=job_id)
            self._send_json(
                {
                    "ok": True,
                    "message": "Pipeline lance.",
                    "job": current,
                },
                status=HTTPStatus.ACCEPTED,
            )
        except RuntimeError as exc:
            if "job_id" in locals():
                shutil.rmtree(Path(tempfile.gettempdir()) / "dms_launcher_uploads" / job_id, ignore_errors=True)
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
        except Exception as exc:
            if "job_id" in locals():
                shutil.rmtree(Path(tempfile.gettempdir()) / "dms_launcher_uploads" / job_id, ignore_errors=True)
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


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
    postgres_status = ensure_postgres_bootstrap(REPO_ROOT, start_if_needed=False)
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
    print(
        "[local-api] postgres "
        f"enabled={1 if postgres_status.get('enabled') else 0} | "
        f"ready={1 if postgres_status.get('ready') else 0} | "
        f"db={postgres_status.get('database')} | "
        f"created={1 if postgres_status.get('database_created') else 0} | "
        f"tables={len(postgres_status.get('tables_ready') or [])}/{len(postgres_status.get('tables_expected') or [])}"
    )
    if postgres_status.get("error"):
        print(f"[local-api] postgres_error={postgres_status.get('error')}")
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
