from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


RUNTIME_STATE_ENV = "DMS_RUNTIME_STATE_PATH"
RUNTIME_JOB_ENV = "DMS_RUNTIME_JOB_ID"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(context: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    raw = None
    if isinstance(context, dict):
        raw = context.get(RUNTIME_STATE_ENV)
    if not raw:
        raw = os.environ.get(RUNTIME_STATE_ENV)
    if not raw:
        return None
    return Path(str(raw)).expanduser()


def _steps(context: Optional[Dict[str, Any]] = None) -> List[str]:
    if not isinstance(context, dict):
        return []
    raw = context.get("PIPELINE_STEPS")
    return [str(item) for item in raw] if isinstance(raw, list) else []


def _pipeline_profile(context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if isinstance(context, dict):
        value = context.get("PIPELINE_PROFILE")
        if value:
            return str(value)
    value = os.environ.get("PIPELINE_PROFILE")
    return str(value) if value else None


def _job_id(context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if isinstance(context, dict):
        value = context.get(RUNTIME_JOB_ENV)
        if value:
            return str(value)
    value = os.environ.get(RUNTIME_JOB_ENV)
    return str(value) if value else None


def read_runtime_state(path_like: str | Path | None) -> Dict[str, Any]:
    if not path_like:
        return {}
    path = Path(path_like)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_runtime_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _step_index(steps: List[str], step_name: Optional[str]) -> int:
    if not step_name or step_name not in steps:
        return 0
    return steps.index(step_name) + 1


def _running_progress_percent(steps_total: int, step_index: int) -> int:
    if steps_total <= 0 or step_index <= 0:
        return 0
    return max(3, min(97, int((((step_index - 1) + 0.5) / steps_total) * 100)))


def update_runtime_state(context: Optional[Dict[str, Any]] = None, **fields: Any) -> Dict[str, Any]:
    path = _state_path(context)
    if path is None:
        return {}

    existing = read_runtime_state(path)
    steps = _steps(context) or list(existing.get("pipeline_steps") or [])
    payload: Dict[str, Any] = {
        **existing,
        "job_id": _job_id(context) or existing.get("job_id"),
        "updated_at": _iso_now(),
        "pipeline_profile": _pipeline_profile(context) or existing.get("pipeline_profile"),
        "pipeline_steps": steps,
        "steps_total": len(steps),
    }
    payload.update(fields)
    _write_runtime_state(path, payload)
    return payload


def publish_pipeline_started(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    steps = _steps(context)
    return update_runtime_state(
        context,
        status="running",
        completed_steps_count=0,
        current_step=None,
        step_index=0,
        progress_percent=0,
        component_name=None,
        component_script=None,
        component_status=None,
        started_at=_iso_now(),
        finished_at=None,
        error=None,
    )


def publish_pipeline_completed(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    steps = _steps(context)
    last_step = steps[-1] if steps else None
    return update_runtime_state(
        context,
        status="completed",
        completed_steps_count=len(steps),
        current_step=last_step,
        step_index=len(steps),
        progress_percent=100,
        component_name=last_step,
        component_status="completed" if last_step else None,
        finished_at=_iso_now(),
        error=None,
    )


def publish_pipeline_failed(context: Optional[Dict[str, Any]] = None, error: Any = None) -> Dict[str, Any]:
    return update_runtime_state(
        context,
        status="failed",
        finished_at=_iso_now(),
        error=str(error) if error else None,
    )


def publish_component_started(
    context: Optional[Dict[str, Any]] = None,
    component_name: Optional[str] = None,
    component_script: Optional[str] = None,
) -> Dict[str, Any]:
    steps = _steps(context)
    step_index = _step_index(steps, component_name)
    return update_runtime_state(
        context,
        status="running",
        current_step=component_name,
        step_index=step_index,
        completed_steps_count=max(0, step_index - 1),
        progress_percent=_running_progress_percent(len(steps), step_index),
        component_name=component_name,
        component_script=component_script,
        component_status="running",
        error=None,
    )


def publish_component_completed(
    context: Optional[Dict[str, Any]] = None,
    component_name: Optional[str] = None,
    component_script: Optional[str] = None,
) -> Dict[str, Any]:
    steps = _steps(context)
    step_index = _step_index(steps, component_name)
    progress_percent = int((step_index / len(steps)) * 100) if steps and step_index else 0
    return update_runtime_state(
        context,
        status="running",
        current_step=component_name,
        step_index=step_index,
        completed_steps_count=step_index,
        progress_percent=progress_percent,
        component_name=component_name,
        component_script=component_script,
        component_status="completed",
    )


def publish_component_failed(
    context: Optional[Dict[str, Any]] = None,
    component_name: Optional[str] = None,
    component_script: Optional[str] = None,
    error: Any = None,
) -> Dict[str, Any]:
    steps = _steps(context)
    step_index = _step_index(steps, component_name)
    return update_runtime_state(
        context,
        status="failed",
        current_step=component_name,
        step_index=step_index,
        completed_steps_count=max(0, step_index - 1),
        progress_percent=_running_progress_percent(len(steps), step_index),
        component_name=component_name,
        component_script=component_script,
        component_status="failed",
        finished_at=_iso_now(),
        error=str(error) if error else None,
    )
