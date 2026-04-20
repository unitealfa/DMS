#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "core" / "main.py"), run_name="__main__")
