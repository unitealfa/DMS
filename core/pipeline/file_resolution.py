from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_LFS_POINTER_RE = re.compile(
    r"^version https://git-lfs\.github\.com/spec/v1\s+"
    r"oid sha256:(?P<oid>[0-9a-f]{64})\s+"
    r"size (?P<size>\d+)\s*$",
    re.DOTALL,
)

_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
}


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_git_lfs_pointer_bytes(data: bytes) -> Optional[Dict[str, Any]]:
    if not data:
        return None
    try:
        text = data.decode("utf-8").strip()
    except Exception:
        return None
    match = _LFS_POINTER_RE.match(text)
    if not match:
        return None
    try:
        return {
            "oid": match.group("oid"),
            "size": int(match.group("size")),
            "text": text,
        }
    except Exception:
        return None


def parse_git_lfs_pointer_path(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    return parse_git_lfs_pointer_bytes(data)


def _iter_named_candidates(root: Path, preferred_name: str) -> Iterable[Path]:
    if not preferred_name or not root.exists():
        return []

    direct = root / preferred_name
    if direct.is_file():
        return [direct]

    matches = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if preferred_name in filenames:
            matches.append(Path(current_root) / preferred_name)
    return matches


def resolve_git_lfs_pointer_path(
    path: Path,
    repo_root: Path,
    preferred_name: str | None = None,
) -> Path:
    pointer = parse_git_lfs_pointer_path(path)
    if not pointer:
        return path
    resolved = _resolve_pointer_spec(pointer, repo_root=repo_root, preferred_name=preferred_name or path.name, avoid_path=path)
    return resolved or path


def _resolve_pointer_spec(
    pointer: Dict[str, Any],
    repo_root: Path,
    preferred_name: str | None = None,
    avoid_path: Path | None = None,
) -> Path | None:
    preferred = str(preferred_name or "").strip()
    search_roots = [repo_root / "documents", repo_root]
    checked: set[str] = set()
    for root in search_roots:
        for candidate in _iter_named_candidates(root, preferred):
            resolved = candidate.resolve()
            key = str(resolved)
            if key in checked:
                continue
            checked.add(key)
            if not resolved.is_file() or (avoid_path is not None and resolved == avoid_path.resolve()):
                continue
            if parse_git_lfs_pointer_path(resolved):
                continue
            try:
                if pointer.get("size") and resolved.stat().st_size != int(pointer["size"]):
                    continue
                if _sha256_path(resolved) != str(pointer["oid"]):
                    continue
            except Exception:
                continue
            return resolved
    return None


def resolve_runtime_input_path(path: Path, repo_root: Path) -> Path:
    try:
        return resolve_git_lfs_pointer_path(path, repo_root=repo_root, preferred_name=path.name)
    except Exception:
        return path


def materialize_uploaded_content_from_lfs_pointer(
    content: bytes,
    repo_root: Path,
    preferred_name: str | None = None,
) -> Dict[str, Any]:
    pointer = parse_git_lfs_pointer_bytes(content)
    if not pointer:
        return {
            "is_lfs_pointer": False,
            "resolved": False,
            "content": content,
            "resolved_source_path": None,
            "oid": None,
            "size": None,
        }

    resolved = _resolve_pointer_spec(pointer, repo_root=repo_root, preferred_name=preferred_name)
    if resolved is not None and resolved.is_file():
        return {
            "is_lfs_pointer": True,
            "resolved": True,
            "content": resolved.read_bytes(),
            "resolved_source_path": str(resolved),
            "oid": pointer.get("oid"),
            "size": pointer.get("size"),
        }

    return {
        "is_lfs_pointer": True,
        "resolved": False,
        "content": content,
        "resolved_source_path": None,
        "oid": pointer.get("oid"),
        "size": pointer.get("size"),
    }
