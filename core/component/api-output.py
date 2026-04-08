from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _read_json(path_like: Any) -> Dict[str, Any]:
    if not path_like:
        return {}
    try:
        path = Path(str(path_like)).expanduser()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path_like: Any, payload: Dict[str, Any]) -> str | None:
    if not path_like:
        return None
    path = Path(str(path_like)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _load_fusion_payload(globals_dict: Dict[str, Any]) -> Dict[str, Any]:
    payload = globals_dict.get("FUSION_PAYLOAD")
    if isinstance(payload, dict):
        return payload
    fusion_path = globals_dict.get("FUSION_RESULT")
    return _read_json(fusion_path)


def _callback_headers(token: str | None = None) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    }
    clean = str(token or "").strip()
    if clean:
        headers["Authorization"] = f"Bearer {clean}"
    return headers


def _deliver_callback(url: str, payload: Dict[str, Any], token: str | None = None, timeout: int = 30) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers=_callback_headers(token), method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed: Any
            try:
                parsed = json.loads(raw) if raw.strip() else None
            except Exception:
                parsed = raw[:1000]
            return {
                "attempted": True,
                "ok": 200 <= int(response.status) < 300,
                "status_code": int(response.status),
                "response": parsed,
                "error": None,
            }
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {
            "attempted": True,
            "ok": False,
            "status_code": int(getattr(exc, "code", 0) or 0),
            "response": raw[:1000],
            "error": str(exc),
        }
    except URLError as exc:
        return {
            "attempted": True,
            "ok": False,
            "status_code": None,
            "response": None,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "attempted": True,
            "ok": False,
            "status_code": None,
            "response": None,
            "error": str(exc),
        }


API_JOB_ID = str(globals().get("API_JOB_ID") or os.environ.get("DMS_API_JOB_ID") or "").strip()
API_MANIFEST_PATH = str(globals().get("API_MANIFEST_PATH") or os.environ.get("DMS_API_MANIFEST_PATH") or "").strip()
API_RESULT_PATH = str(globals().get("API_RESULT_PATH") or os.environ.get("DMS_API_RESULT_PATH") or "").strip()
API_RESULT_ROUTE = str(globals().get("API_RESULT_ROUTE") or os.environ.get("DMS_API_RESULT_ROUTE") or "").strip()
API_RESULT_URL = str(globals().get("API_RESULT_URL") or os.environ.get("DMS_API_RESULT_URL") or "").strip()
API_CALLBACK_URL = str(globals().get("API_CALLBACK_URL") or os.environ.get("DMS_API_CALLBACK_URL") or "").strip()
API_CALLBACK_TOKEN = str(globals().get("API_CALLBACK_TOKEN") or os.environ.get("DMS_API_CALLBACK_TOKEN") or "").strip()
API_CALLBACK_TIMEOUT = int(str(globals().get("API_CALLBACK_TIMEOUT") or os.environ.get("DMS_API_CALLBACK_TIMEOUT") or "30") or "30")

PIPELINE_PROFILE = str(globals().get("PIPELINE_PROFILE") or os.environ.get("PIPELINE_PROFILE") or "").strip() or None
PIPELINE_STEPS = _safe_list(globals().get("PIPELINE_STEPS"))

fusion_payload = _load_fusion_payload(globals())
manifest = _read_json(API_MANIFEST_PATH)
stored_documents = _safe_list(_safe_dict(manifest).get("documents"))

api_output_payload: Dict[str, Any] = {
    "ok": True,
    "schema_version": "api-output-1.0",
    "generated_at": _iso_now(),
    "job_id": API_JOB_ID or None,
    "pipeline_profile": PIPELINE_PROFILE,
    "pipeline_steps": PIPELINE_STEPS,
    "source": "local-api",
    "input_documents_count": len(stored_documents),
    "input_documents": stored_documents,
    "manifest": manifest if manifest else None,
    # Le payload fusionne est recopie tel quel, sans reduction.
    "pipeline_output": fusion_payload,
}

callback_report = {
    "attempted": False,
    "ok": None,
    "status_code": None,
    "response": None,
    "error": None,
    "url": API_CALLBACK_URL or None,
}

if API_CALLBACK_URL:
    callback_report = _deliver_callback(
        API_CALLBACK_URL,
        api_output_payload,
        token=API_CALLBACK_TOKEN or None,
        timeout=API_CALLBACK_TIMEOUT,
    )
    callback_report["url"] = API_CALLBACK_URL

api_output_payload["callback_delivery"] = callback_report

written_path = _write_json(API_RESULT_PATH, api_output_payload)
api_output_payload["result_path"] = written_path
api_output_payload["result_route"] = API_RESULT_ROUTE or None
api_output_payload["result_url"] = API_RESULT_URL or None

if manifest:
    current_result_meta = _safe_dict(manifest.get("result"))
    manifest["result"] = {
        "ready": bool(written_path),
        "generated_at": api_output_payload["generated_at"],
        "path": written_path,
        "route": API_RESULT_ROUTE or current_result_meta.get("route"),
        "url": API_RESULT_URL or current_result_meta.get("url"),
        "documents_count": len(_safe_list(_safe_dict(fusion_payload).get("documents"))),
        "pipeline_profile": PIPELINE_PROFILE,
    }
    manifest["callback"] = {
        "url": API_CALLBACK_URL or None,
        "attempted": bool(callback_report.get("attempted")),
        "ok": callback_report.get("ok"),
        "status_code": callback_report.get("status_code"),
        "error": callback_report.get("error"),
    }
    _write_json(API_MANIFEST_PATH, manifest)

API_OUTPUT_RESULT = {
    "job_id": API_JOB_ID or None,
    "pipeline_profile": PIPELINE_PROFILE,
    "documents_count": len(_safe_list(_safe_dict(fusion_payload).get("documents"))),
    "result_path": written_path,
    "result_route": API_RESULT_ROUTE or None,
    "result_url": API_RESULT_URL or None,
    "callback_attempted": bool(callback_report.get("attempted")),
    "callback_ok": callback_report.get("ok"),
}

print(
    "[api-output] "
    f"job={API_JOB_ID or '-'} | profile={PIPELINE_PROFILE or '-'} | "
    f"docs={API_OUTPUT_RESULT['documents_count']} | "
    f"result_path={written_path or '-'} | "
    f"callback={'yes' if API_CALLBACK_URL else 'no'}"
)
