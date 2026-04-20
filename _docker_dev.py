#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
API_CONTAINER = "dms-api"
ELASTICSEARCH_CONTAINER = "dms-elasticsearch"
DEFAULT_API_URL = "http://127.0.0.1:8765"
DEFAULT_LOCAL_API_ARGS = ["--host", "0.0.0.0", "--port", "8765"]
MIN_LOCAL_MODULES = [
    "PIL",
    "pytesseract",
    "numpy",
    "requests",
    "yaml",
    "elasticsearch",
    "openpyxl",
    "langdetect",
]


def _is_truthy(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def prefer_docker() -> bool:
    if _is_truthy(os.environ.get("DMS_FORCE_LOCAL")):
        return False
    if _is_truthy(os.environ.get("DMS_FORCE_DOCKER")):
        return True
    return not local_runtime_ready()


def local_runtime_ready() -> bool:
    for module_name in MIN_LOCAL_MODULES:
        if importlib.util.find_spec(module_name) is None:
            return False
    return True


def docker_base_command() -> list[str] | None:
    candidates = (["docker"], ["sudo", "docker"])
    for candidate in candidates:
        try:
            probe = subprocess.run(
                [*candidate, "ps"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            continue
        if probe.returncode == 0:
            return list(candidate)
    return None


def _run(cmd: Sequence[str], *, cwd: Path | None = None, check: bool = True) -> int:
    completed = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=False)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return int(completed.returncode)


def _container_exists(base: Sequence[str], name: str) -> bool:
    completed = subprocess.run(
        [*base, "container", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _container_running(base: Sequence[str], name: str) -> bool:
    completed = subprocess.run(
        [*base, "inspect", "-f", "{{.State.Running}}", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip().lower() == "true"


def ensure_stack_started() -> list[str]:
    base = docker_base_command()
    if not base:
        raise SystemExit("Docker n'est pas disponible. Installe Docker ou utilise l'environnement Python local.")
    build = not _container_exists(base, API_CONTAINER)
    command = [*base, "compose", "up", "-d"]
    if build:
        command.append("--build")
    command.extend(["elasticsearch", "dms-api"])
    _run(command, cwd=ROOT, check=True)
    return base


def docker_exec_python(script_name: str, script_args: Sequence[str]) -> int:
    base = ensure_stack_started()
    command = [*base, "exec"]
    if sys.stdin.isatty():
        command.append("-i")
    command.extend([API_CONTAINER, "python", script_name, *script_args])
    return _run(command, check=False)


def restart_api_service(script_args: Sequence[str] | None = None) -> int:
    base = ensure_stack_started()
    custom_args = list(script_args or [])
    if custom_args and custom_args != DEFAULT_LOCAL_API_ARGS:
        print(
            "[docker-wrapper] Les arguments custom de local_api.py ne sont pas reappliques au service Docker en cours.",
            file=sys.stderr,
        )
        print(
            f"[docker-wrapper] Le service dms-api reste expose sur {DEFAULT_API_URL} selon docker-compose.yml.",
            file=sys.stderr,
        )
    if _container_running(base, API_CONTAINER):
        _run([*base, "restart", API_CONTAINER], check=True)
    print(f"[docker-wrapper] API Docker active sur {DEFAULT_API_URL}")
    if sys.stdout.isatty():
        return _run([*base, "logs", "-f", "--tail", "120", API_CONTAINER], check=False)
    return 0
