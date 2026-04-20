#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys

from _docker_dev import CORE, prefer_docker, restart_api_service


def _run_local() -> None:
    sys.path.insert(0, str(CORE))
    runpy.run_path(str(CORE / "local_api.py"), run_name="__main__")


if __name__ == "__main__":
    if prefer_docker():
        raise SystemExit(restart_api_service(sys.argv[1:]))
    try:
        _run_local()
    except (ModuleNotFoundError, ImportError):
        raise SystemExit(restart_api_service(sys.argv[1:]))
