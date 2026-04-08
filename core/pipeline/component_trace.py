from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_component_key(name: Any) -> str:
    value = str(name or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "component"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _lines(value: Any) -> List[str]:
    text = _text(value)
    return text.splitlines() if text else []


def _rebuild_terminal_snapshot(trace: Dict[str, Any]) -> None:
    execution_line = _text(trace.get("execution_line")).strip()
    stdout_text = _text(trace.get("stdout_text"))
    stderr_text = _text(trace.get("stderr_text"))
    report_text = _text(trace.get("report_text"))

    segments = []
    if execution_line:
        segments.append(execution_line)
    if stdout_text:
        segments.append(stdout_text.rstrip("\n"))
    if stderr_text:
        segments.append(stderr_text.rstrip("\n"))
    if report_text:
        segments.append(report_text.rstrip("\n"))

    terminal_text = "\n".join(segment for segment in segments if segment)
    trace["stdout_lines"] = _lines(stdout_text)
    trace["stderr_lines"] = _lines(stderr_text)
    trace["report_lines"] = _lines(report_text)
    trace["terminal_text"] = terminal_text or None
    trace["terminal_lines"] = _lines(terminal_text)


def _fingerprint(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return (type(value).__name__, value)
    if isinstance(value, str):
        preview = value[:200] if len(value) > 200 else value
        return ("str", len(value), preview)
    if isinstance(value, dict):
        keys = sorted(str(k) for k in value.keys())
        return ("dict", len(value), tuple(keys[:30]))
    if isinstance(value, list):
        first_type = type(value[0]).__name__ if value else None
        return ("list", len(value), first_type)
    if isinstance(value, tuple):
        first_type = type(value[0]).__name__ if value else None
        return ("tuple", len(value), first_type)
    if isinstance(value, set):
        return ("set", len(value))
    return (type(value).__name__, id(value))


def capture_context_fingerprints(context: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in context.items():
        if str(key).startswith("__"):
            continue
        out[str(key)] = _fingerprint(value)
    return out


def _ensure_component_traces(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = context.get("COMPONENT_TRACES")
    if not isinstance(rows, list):
        rows = []
        context["COMPONENT_TRACES"] = rows
    return rows


def _pipeline_step_index(context: Dict[str, Any], component_name: str) -> int:
    steps = context.get("PIPELINE_STEPS")
    if not isinstance(steps, list):
        return 0
    try:
        return [str(x) for x in steps].index(component_name) + 1
    except Exception:
        return 0


def start_component_trace(context: Dict[str, Any], component_name: str, component_script: str) -> Dict[str, Any]:
    traces = _ensure_component_traces(context)
    trace = {
        "component_name": component_name,
        "component_key": sanitize_component_key(component_name),
        "component_script": str(component_script or ""),
        "execution_line": f"Execution du composant {component_name} via {component_script}",
        "step_index": _pipeline_step_index(context, component_name),
        "status": "running",
        "started_at": _iso_now(),
        "finished_at": None,
        "new_context_keys": [],
        "changed_context_keys": [],
        "context_keys_touched": [],
        "summary": None,
        "reported_output_type": None,
        "reported_output": None,
        "stdout_text": "",
        "stdout_lines": [],
        "stderr_text": "",
        "stderr_lines": [],
        "report_text": None,
        "report_lines": [],
        "terminal_text": None,
        "terminal_lines": [],
        "error": None,
    }
    traces.append(trace)
    return trace


def finish_component_trace(
    trace: Optional[Dict[str, Any]],
    before_fingerprints: Dict[str, Any],
    context: Dict[str, Any],
    *,
    status: str,
    error: Any = None,
) -> None:
    if not isinstance(trace, dict):
        return
    after_fingerprints = capture_context_fingerprints(context)
    before_keys = set(before_fingerprints.keys())
    after_keys = set(after_fingerprints.keys())
    new_keys = sorted(after_keys - before_keys)
    changed_keys = sorted(
        key for key in (before_keys & after_keys) if before_fingerprints.get(key) != after_fingerprints.get(key)
    )
    trace["status"] = status
    trace["finished_at"] = _iso_now()
    trace["new_context_keys"] = new_keys
    trace["changed_context_keys"] = changed_keys
    trace["context_keys_touched"] = sorted(set(new_keys + changed_keys))
    trace["error"] = str(error) if error else None


def report_component_trace(
    trace: Optional[Dict[str, Any]],
    *,
    output: Any,
    summary: str,
) -> None:
    if not isinstance(trace, dict):
        return
    trace["summary"] = str(summary or "")
    trace["reported_output_type"] = type(output).__name__
    # On garde un snapshot JSON-safe pour les composants futurs qui ne seraient
    # pas encore explicitement fusionnes.
    trace["reported_output"] = _json_safe(output)
    if not trace.get("finished_at"):
        trace["finished_at"] = _iso_now()
    if trace.get("status") == "running":
        trace["status"] = "completed"


def attach_component_io(
    trace: Optional[Dict[str, Any]],
    *,
    stdout_text: Any = "",
    stderr_text: Any = "",
) -> None:
    if not isinstance(trace, dict):
        return
    trace["stdout_text"] = _text(stdout_text)
    trace["stderr_text"] = _text(stderr_text)
    _rebuild_terminal_snapshot(trace)


def attach_component_report(trace: Optional[Dict[str, Any]], report_lines: List[str]) -> None:
    if not isinstance(trace, dict):
        return
    clean_lines = [str(line) for line in (report_lines or [])]
    trace["report_text"] = "\n".join(clean_lines) if clean_lines else None
    _rebuild_terminal_snapshot(trace)


def component_trace_public_rows(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = context.get("COMPONENT_TRACES")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "component_name": row.get("component_name"),
                "component_key": row.get("component_key"),
                "component_script": row.get("component_script"),
                "execution_line": row.get("execution_line"),
                "step_index": _safe_int(row.get("step_index"), 0),
                "status": row.get("status"),
                "started_at": row.get("started_at"),
                "finished_at": row.get("finished_at"),
                "summary": row.get("summary"),
                "new_context_keys": list(row.get("new_context_keys") or []),
                "changed_context_keys": list(row.get("changed_context_keys") or []),
                "context_keys_touched": list(row.get("context_keys_touched") or []),
                "reported_output_type": row.get("reported_output_type"),
                "reported_output": row.get("reported_output"),
                "stdout_text": row.get("stdout_text"),
                "stdout_lines": list(row.get("stdout_lines") or []),
                "stderr_text": row.get("stderr_text"),
                "stderr_lines": list(row.get("stderr_lines") or []),
                "report_text": row.get("report_text"),
                "report_lines": list(row.get("report_lines") or []),
                "terminal_text": row.get("terminal_text"),
                "terminal_lines": list(row.get("terminal_lines") or []),
                "error": row.get("error"),
            }
        )
    return out
