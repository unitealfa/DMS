from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.postgres import (  # noqa: E402
    attach_postgres_sync_audit,
    ensure_postgres_bootstrap,
    load_postgres_connection_config,
    sync_fusion_payload_to_postgres,
)


DEFAULT_FUSION_PATH = REPO_ROOT / "fusion_output.json"


def _load_payload_from_disk(path: Path) -> Dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def main() -> None:
    ctx = globals()
    cfg = load_postgres_connection_config(REPO_ROOT)
    payload = ctx.get("FUSION_PAYLOAD")
    fusion_path = Path(str(ctx.get("FUSION_RESULT") or DEFAULT_FUSION_PATH)).expanduser()
    if not fusion_path.is_absolute():
        fusion_path = (REPO_ROOT / fusion_path).resolve()

    if not isinstance(payload, dict) and fusion_path.exists():
        payload = _load_payload_from_disk(fusion_path)

    bootstrap_status = ensure_postgres_bootstrap(REPO_ROOT, start_if_needed=True)

    sync_status = sync_fusion_payload_to_postgres(
        REPO_ROOT,
        payload,
        bootstrap_status=bootstrap_status,
        run_id=ctx.get("PIPELINE_RUN_ID"),
        pipeline_profile=ctx.get("PIPELINE_PROFILE"),
        source=ctx.get("FUSION_SOURCE"),
        fusion_path=fusion_path,
    )

    ctx["POSTGRES_STATUS"] = bootstrap_status
    ctx["POSTGRES_ENABLED"] = bool(bootstrap_status.get("enabled"))
    ctx["POSTGRES_READY"] = bool(bootstrap_status.get("ready"))
    ctx["POSTGRES_HOST"] = bootstrap_status.get("host")
    ctx["POSTGRES_PORT"] = bootstrap_status.get("port")
    ctx["POSTGRES_USER"] = bootstrap_status.get("user")
    ctx["POSTGRES_DATABASE"] = bootstrap_status.get("database")
    ctx["POSTGRES_TABLES"] = bootstrap_status.get("tables_expected") or []
    ctx["POSTGRES_SYNC"] = sync_status
    if sync_status.get("run_id"):
        ctx["PIPELINE_RUN_ID"] = sync_status["run_id"]

    if isinstance(payload, dict) and cfg.sync_write_fusion_audit:
        payload = attach_postgres_sync_audit(payload, sync_status)
        ctx["FUSION_PAYLOAD"] = payload
        if fusion_path:
            fusion_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            sync_status["fusion_audit_written"] = True
            ctx["FUSION_RESULT"] = str(fusion_path)

    print(
        "[postgres-sync] "
        f"enabled={1 if sync_status.get('enabled') else 0} | "
        f"ready={1 if sync_status.get('ready') else 0} | "
        f"run={sync_status.get('run_id')} | "
        f"docs={int(sync_status.get('documents_upserted') or 0)}/{int(sync_status.get('documents_total') or 0)} | "
        f"payloads={int(sync_status.get('payloads_upserted') or 0)} | "
        f"nodes={int(sync_status.get('run_payload_nodes_upserted') or 0) + int(sync_status.get('document_payload_nodes_upserted') or 0)} | "
        f"ids={int(sync_status.get('identifiers_upserted') or 0)} | "
        f"pages={int(sync_status.get('pages_upserted') or 0)} | "
        f"tokens={int(sync_status.get('tokens_upserted') or 0)} | "
        f"topics={int(sync_status.get('topics_upserted') or 0)} | "
        f"extracts={int(sync_status.get('extraction_details_upserted') or 0)} | "
        f"tables={int(sync_status.get('tables_upserted') or 0)} | "
        f"rows={int(sync_status.get('table_rows_upserted') or 0)} | "
        f"cells={int(sync_status.get('table_cells_upserted') or 0)} | "
        f"links={int(sync_status.get('links_upserted') or 0)} | "
        f"registry={int(sync_status.get('stable_registry_upserted') or 0)} | "
        f"skipped={sync_status.get('skipped') or 'no'}"
    )
    if sync_status.get("error"):
        print(f"[postgres-sync][error] {sync_status.get('error')}")


if __name__ == "__main__":
    main()
