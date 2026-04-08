from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pipeline.component_trace import component_trace_public_rows, sanitize_component_key


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_OUTPUT_PATH = REPO_ROOT / "dms-unified-output-template.json"


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


def _load_template_payload() -> Dict[str, Any]:
    return _read_json(TEMPLATE_OUTPUT_PATH)


def _materialize_raw(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _materialize_raw(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_materialize_raw(v) for v in value]
    return value


def _same_filename(a: Any, b: Any) -> bool:
    try:
        return Path(str(a or "")).name.strip().lower() == Path(str(b or "")).name.strip().lower()
    except Exception:
        return False


def _row_belongs_to_doc(row: Any, doc_id: str | None, filename: str | None) -> bool:
    if not isinstance(row, dict):
        return False
    row_doc_id = str(row.get("doc_id") or row.get("document_id") or "").strip()
    row_filename = str(row.get("filename") or row.get("doc") or "").strip()
    if doc_id and row_doc_id and row_doc_id == doc_id:
        return True
    if filename and row_filename and (row_filename == filename or _same_filename(row_filename, filename)):
        return True
    return False


def _filter_rows_for_doc(rows: Any, doc_id: str | None, filename: str | None) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [row for row in rows if _row_belongs_to_doc(row, doc_id, filename)]


def _pick_dynamic_template(template: Dict[str, Any]) -> Any:
    for key in (
        "_dynamic_field_template",
        "_dynamic_doc_type_template",
        "_dynamic_language_template",
        "_dynamic_duration_template",
    ):
        if key in template:
            return template.get(key)
    if "_dynamic_fields" in template:
        return {}
    return None


def _merge_template(template: Any, actual: Any) -> Any:
    if isinstance(template, dict):
        result: Dict[str, Any] = {}
        actual_dict = actual if isinstance(actual, dict) else {}

        for key, template_value in template.items():
            if key.startswith("_dynamic_"):
                result[key] = _merge_template(template_value, None)
                continue
            result[key] = _merge_template(template_value, actual_dict.get(key))

        dynamic_template = _pick_dynamic_template(template)
        for key, actual_value in actual_dict.items():
            if key in result:
                continue
            if dynamic_template is not None:
                result[key] = _merge_template(dynamic_template, actual_value)
            else:
                result[key] = _materialize_raw(actual_value)
        return result

    if isinstance(template, list):
        item_template = template[0] if template else None
        if isinstance(actual, list):
            if not actual:
                return []
            if item_template is None:
                return [_materialize_raw(item) for item in actual]
            return [_merge_template(item_template, item) for item in actual]
        if actual is None:
            return []
        if item_template is None:
            return [_materialize_raw(actual)]
        return [_merge_template(item_template, actual)]

    if actual is None:
        return template if template is not None else None
    return actual


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


def _stored_input_files(stored_documents: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for item in stored_documents:
        if not isinstance(item, dict):
            continue
        for key in ("stored_absolute_path", "stored_relative_path", "file_name"):
            value = str(item.get(key) or "").strip()
            if value:
                out.append(value)
                break
    return out


def _normalize_for_api(
    fusion_payload: Dict[str, Any],
    template_payload: Dict[str, Any],
    *,
    pipeline_profile: str | None,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = _merge_template(template_payload, fusion_payload)
    stored_documents = _safe_list(_safe_dict(manifest).get("documents"))
    effective_profile = (
        str(fusion_payload.get("profile") or "").strip()
        or str(pipeline_profile or "").strip()
        or None
    )

    normalized["schema_version"] = template_payload.get("schema_version") or normalized.get("schema_version")
    normalized["generated_at"] = _iso_now()
    if effective_profile:
        normalized["profile"] = effective_profile
    if "source" in fusion_payload and fusion_payload.get("source") is not None:
        normalized["source"] = fusion_payload.get("source")
    normalized["documents_count"] = len(_safe_list(normalized.get("documents")))

    pipeline_obj = _safe_dict(normalized.get("pipeline"))
    if effective_profile:
        pipeline_obj["profile"] = effective_profile
    normalized["pipeline"] = pipeline_obj

    source_context = _safe_dict(normalized.get("source_context"))
    manifest_path = str(API_MANIFEST_PATH or "").strip()
    source_context["corpus_root"] = str(Path(manifest_path).expanduser().parent) if manifest_path else source_context.get("corpus_root")
    source_context["input_files"] = _stored_input_files(stored_documents)
    source_context["source_mode"] = "api"
    source_context["fusion_source"] = fusion_payload.get("source")
    source_context["fusion_schema_version"] = fusion_payload.get("schema_version")
    source_context["label"] = "local-api"
    source_context["profile_requested"] = pipeline_profile
    source_context["profile_effective"] = effective_profile
    normalized["source_context"] = source_context
    return normalized


def _component_trace_context_values(globals_dict: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in trace.get("context_keys_touched") or []:
        if key in globals_dict:
            out[str(key)] = globals_dict.get(key)
    return out


def _extract_doc_value_from_any(value: Any, doc_id: str | None, filename: str | None, docs_total: int) -> Any:
    if isinstance(value, dict):
        if _row_belongs_to_doc(value, doc_id, filename):
            return value
        if docs_total == 1:
            return value
        return None
    if isinstance(value, list):
        matched = _filter_rows_for_doc(value, doc_id, filename)
        if matched:
            return matched if len(matched) != 1 else matched[0]
        if docs_total == 1 and value and not any(isinstance(item, dict) for item in value):
            return value
        return None
    if docs_total == 1:
        return value
    return None


def _overlay_component_traces(
    globals_dict: Dict[str, Any],
    normalized_payload: Dict[str, Any],
) -> Dict[str, Any]:
    traces = [row for row in component_trace_public_rows(globals_dict) if isinstance(row, dict)]
    if not traces:
        return normalized_payload

    pipeline_obj = _safe_dict(normalized_payload.get("pipeline"))
    existing_runs = _safe_list(pipeline_obj.get("component_runs"))
    if not existing_runs:
        pipeline_obj["component_runs"] = traces
    else:
        seen = {(str(row.get("component_key") or ""), str(row.get("step_index") or "")) for row in existing_runs if isinstance(row, dict)}
        for trace in traces:
            ident = (str(trace.get("component_key") or ""), str(trace.get("step_index") or ""))
            if ident in seen:
                continue
            existing_runs.append(trace)
        pipeline_obj["component_runs"] = existing_runs
    normalized_payload["pipeline"] = pipeline_obj

    docs = _safe_list(normalized_payload.get("documents"))
    docs_total = len(docs)
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        components = _safe_dict(doc.get("components"))
        doc_id = str(doc.get("document_id") or "").strip() or None
        file_obj = _safe_dict(doc.get("file"))
        filename = str(file_obj.get("name") or "").strip() or None
        for trace in traces:
            comp_key = sanitize_component_key(trace.get("component_name"))
            if comp_key in components:
                continue
            data: Dict[str, Any] = {}
            for key, value in _component_trace_context_values(globals_dict, trace).items():
                extracted = _extract_doc_value_from_any(value, doc_id, filename, docs_total)
                if extracted is not None:
                    data[key] = extracted
            if not data and docs_total == 1 and trace.get("reported_output") is not None:
                data["reported_output"] = trace.get("reported_output")
            if not data:
                continue
            components[comp_key] = {
                "component_name": trace.get("component_name"),
                "script": trace.get("component_script"),
                "status": trace.get("status"),
                "summary": trace.get("summary"),
                "context_keys": trace.get("context_keys_touched") or [],
                "output_type": trace.get("reported_output_type"),
                "data": _materialize_raw(data),
            }
        doc["components"] = components
    normalized_payload["documents"] = docs
    return normalized_payload


API_JOB_ID = str(globals().get("API_JOB_ID") or os.environ.get("DMS_API_JOB_ID") or "").strip()
API_MANIFEST_PATH = str(globals().get("API_MANIFEST_PATH") or os.environ.get("DMS_API_MANIFEST_PATH") or "").strip()
API_RESULT_PATH = str(globals().get("API_RESULT_PATH") or os.environ.get("DMS_API_RESULT_PATH") or "").strip()
API_RESULT_ROUTE = str(globals().get("API_RESULT_ROUTE") or os.environ.get("DMS_API_RESULT_ROUTE") or "").strip()
API_RESULT_URL = str(globals().get("API_RESULT_URL") or os.environ.get("DMS_API_RESULT_URL") or "").strip()
API_CALLBACK_URL = str(globals().get("API_CALLBACK_URL") or os.environ.get("DMS_API_CALLBACK_URL") or "").strip()
API_CALLBACK_TOKEN = str(globals().get("API_CALLBACK_TOKEN") or os.environ.get("DMS_API_CALLBACK_TOKEN") or "").strip()
API_CALLBACK_TIMEOUT = int(str(globals().get("API_CALLBACK_TIMEOUT") or os.environ.get("DMS_API_CALLBACK_TIMEOUT") or "30") or "30")

PIPELINE_PROFILE = str(globals().get("PIPELINE_PROFILE") or os.environ.get("PIPELINE_PROFILE") or "").strip() or None

fusion_payload = _load_fusion_payload(globals())
template_payload = _load_template_payload()
manifest = _read_json(API_MANIFEST_PATH)
stored_documents = _safe_list(_safe_dict(manifest).get("documents"))

api_result_payload = _normalize_for_api(
    fusion_payload,
    template_payload,
    pipeline_profile=PIPELINE_PROFILE,
    manifest=manifest,
)
api_result_payload = _overlay_component_traces(globals(), api_result_payload)

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
        api_result_payload,
        token=API_CALLBACK_TOKEN or None,
        timeout=API_CALLBACK_TIMEOUT,
    )
    callback_report["url"] = API_CALLBACK_URL

written_path = _write_json(API_RESULT_PATH, api_result_payload)

if manifest:
    current_result_meta = _safe_dict(manifest.get("result"))
    manifest["result"] = {
        "ready": bool(written_path),
        "generated_at": api_result_payload.get("generated_at"),
        "path": written_path,
        "route": API_RESULT_ROUTE or current_result_meta.get("route"),
        "url": API_RESULT_URL or current_result_meta.get("url"),
        "documents_count": len(_safe_list(_safe_dict(api_result_payload).get("documents"))),
        "pipeline_profile": PIPELINE_PROFILE or api_result_payload.get("profile"),
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
    "pipeline_profile": PIPELINE_PROFILE or api_result_payload.get("profile"),
    "documents_count": len(_safe_list(_safe_dict(api_result_payload).get("documents"))),
    "result_path": written_path,
    "result_route": API_RESULT_ROUTE or None,
    "result_url": API_RESULT_URL or None,
    "callback_attempted": bool(callback_report.get("attempted")),
    "callback_ok": callback_report.get("ok"),
    "result_schema_version": api_result_payload.get("schema_version"),
}

print(
    "[api-output] "
    f"job={API_JOB_ID or '-'} | profile={API_OUTPUT_RESULT['pipeline_profile'] or '-'} | "
    f"docs={API_OUTPUT_RESULT['documents_count']} | "
    f"schema={API_OUTPUT_RESULT['result_schema_version'] or '-'} | "
    f"result_path={written_path or '-'} | "
    f"callback={'yes' if API_CALLBACK_URL else 'no'}"
)
