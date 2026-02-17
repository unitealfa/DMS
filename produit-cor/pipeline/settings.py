from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Sequence, Union

InputLike = Union[str, Sequence[str], None]
Context = Dict[str, Any]

# Root of the repository (one level above this file)
REPO_ROOT = Path(__file__).resolve().parent.parent
COMPONENT_DIR = REPO_ROOT / "component"
LOG_PATH = REPO_ROOT / "orchestre.log"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def normalize_input(input_data: InputLike) -> List[str]:
    """Return a clean list of file paths from user input."""
    if input_data is None:
        return []

    def _split_item(item: str) -> List[str]:
        parts = [p.strip() for p in item.replace("\\", "/").split(",") if p.strip()]
        return parts or ([item.strip()] if item.strip() else [])

    if isinstance(input_data, str):
        return _split_item(input_data)

    out: List[str] = []
    for elem in input_data:
        out.extend(_split_item(str(elem)))
    return out


def safe_repr(obj: Any) -> str:
    """Pretty-print without truncation to keep auditability."""
    return pformat(obj, width=100, compact=True)


def count_sentences(tok_docs: Any) -> int:
    total = 0
    for doc in tok_docs or []:
        for pg in doc.get("pages", []) or []:
            total += len(pg.get("sentences_layout") or [])
    return total


@contextmanager
def change_dir(path: Path) -> Iterator[None]:
    """Temporarily change working directory."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextmanager
def isolated_argv(argv: Sequence[str]) -> Iterator[None]:
    """Temporarily replace sys.argv so notebook-style scripts parse cleanly."""
    original = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original
