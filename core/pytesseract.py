from __future__ import annotations

import csv
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image


class TesseractError(RuntimeError):
    pass


class TesseractNotFoundError(TesseractError):
    pass


class Output:
    DICT = "dict"


_TESSERACT_BIN = str(os.environ.get("TESSERACT_CMD") or "tesseract")


def _split_config(config: str) -> List[str]:
    raw = str(config or "").strip()
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=(os.name != "nt"))
    except Exception:
        return raw.split()


def _run_tesseract(args: Sequence[str]) -> str:
    cmd = [_TESSERACT_BIN, *args]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise TesseractNotFoundError(f"Tesseract binary not found: {_TESSERACT_BIN}") from exc

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip() or f"Tesseract failed with exit code {proc.returncode}"
        raise TesseractError(message)

    return proc.stdout if proc.stdout else proc.stderr


def _coerce_image_input(image: Any) -> Tuple[str, bool]:
    if isinstance(image, (str, Path)):
        return str(Path(image)), False
    if isinstance(image, Image.Image):
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = handle.name
        handle.close()
        image.save(tmp_path, format="PNG")
        return tmp_path, True
    raise TesseractError(f"Unsupported image input type: {type(image)!r}")


def _cleanup_temp(path: str, delete_after: bool) -> None:
    if not delete_after:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def get_tesseract_version() -> str:
    output = _run_tesseract(["--version"])
    for line in output.splitlines():
        line = line.strip()
        if line:
            return line
    return output.strip()


def get_languages(config: str = "") -> List[str]:
    output = _run_tesseract(["--list-langs", *_split_config(config)])
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("list of available languages"):
        lines = lines[1:]
    return lines


def image_to_osd(image: Any, config: str = "") -> str:
    source_path, should_delete = _coerce_image_input(image)
    try:
        args = [source_path, "stdout", "-l", "osd", "--psm", "0", *_split_config(config)]
        return _run_tesseract(args)
    finally:
        _cleanup_temp(source_path, should_delete)


def image_to_string(image: Any, lang: str | None = None, config: str = "") -> str:
    source_path, should_delete = _coerce_image_input(image)
    try:
        args = [source_path, "stdout"]
        if lang:
            args.extend(["-l", str(lang)])
        args.extend(_split_config(config))
        return _run_tesseract(args)
    finally:
        _cleanup_temp(source_path, should_delete)


def _parse_tsv_to_dict(raw_tsv: str) -> Dict[str, List[str]]:
    rows = list(csv.DictReader(raw_tsv.splitlines(), delimiter="\t"))
    if not rows:
        return {}

    out: Dict[str, List[str]] = {}
    fieldnames = list(rows[0].keys())
    for field in fieldnames:
        if field is None:
            continue
        out[field] = [str(row.get(field, "") or "") for row in rows]
    return out


def image_to_data(
    image: Any,
    lang: str | None = None,
    config: str = "",
    output_type: Any = None,
) -> Dict[str, List[str]]:
    source_path, should_delete = _coerce_image_input(image)
    try:
        args = [source_path, "stdout"]
        if lang:
            args.extend(["-l", str(lang)])
        args.extend(_split_config(config))
        args.append("tsv")
        result = _parse_tsv_to_dict(_run_tesseract(args))
        if output_type not in (None, Output.DICT, "dict"):
            raise TesseractError(f"Unsupported output_type: {output_type!r}")
        return result
    finally:
        _cleanup_temp(source_path, should_delete)
