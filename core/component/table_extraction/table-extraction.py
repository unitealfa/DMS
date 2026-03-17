from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from table_extraction_lib import run_table_extraction


def _resolve_profile(ctx: Dict[str, Any]) -> str:
    raw = str(ctx.get("PIPELINE_PROFILE") or "").strip().lower()
    if "100" in raw:
        return "100ml"
    if "50" in raw:
        return "50ml"
    return "100ml"


_CTX = globals()
_PROFILE = _resolve_profile(_CTX)
TABLE_EXTRACTIONS = run_table_extraction(_CTX, profile=_PROFILE)

if _PROFILE == "50ml":
    TABLE_EXTRACTIONS_50ML = TABLE_EXTRACTIONS
elif _PROFILE == "100ml":
    TABLE_EXTRACTIONS_100ML = TABLE_EXTRACTIONS
