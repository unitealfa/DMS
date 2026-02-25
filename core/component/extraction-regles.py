"""
Application des règles regex selon la classification.
- Récupère le meilleur doc_type depuis `RESULTS` produit par component/clasification.py.
- Sélectionne les règles à appliquer via config/ruleset_routes.json (include_common, mapping, default_rulesets).
- Charge dynamiquement les fichiers de règles présents dans le dossier rules/.
- Parcourt les textes des documents (TOK_DOCS > FINAL_DOCS > DOCS > TEXT_DOCS) et applique les regex.
- Expose EXTRACTIONS pour le pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Dossiers
REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_DIR = REPO_ROOT / "rules"
CONFIG_DIR = REPO_ROOT / "config"
ROUTES_PATH = CONFIG_DIR / "ruleset_routes.json"


# ----------------- Helpers -----------------
def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _compile_regex(pattern: str, flags_list: Optional[List[str]]) -> re.Pattern:
    flags = 0
    for f in flags_list or []:
        f2 = (f or "").upper()
        if f2 == "IGNORECASE":
            flags |= re.IGNORECASE
        elif f2 == "MULTILINE":
            flags |= re.MULTILINE
        elif f2 == "DOTALL":
            flags |= re.DOTALL
        elif f2 == "VERBOSE":
            flags |= re.VERBOSE
    return re.compile(pattern, flags)


# ----------------- Routes -----------------
def _load_routes() -> Dict[str, Any]:
    default = {
        "include_common": True,
        "default_rulesets": ["{doc_type}.json"],
        "ruleset_mapping": {},
    }
    if not ROUTES_PATH.exists():
        return default
    cfg = _load_json(ROUTES_PATH)
    if not isinstance(cfg, dict):
        return default
    cfg.setdefault("include_common", True)
    cfg.setdefault("default_rulesets", ["{doc_type}.json"])
    cfg.setdefault("ruleset_mapping", {})
    return cfg


def _resolve_rulesets(doc_type: str, routes: Dict[str, Any]) -> List[str]:
    doc_type_up = (doc_type or "").upper()
    mapping: Dict[str, List[str]] = routes.get("ruleset_mapping", {}) or {}
    if doc_type_up in mapping:
        return list(mapping[doc_type_up])

    out: List[str] = []
    for tpl in routes.get("default_rulesets", []) or []:
        try:
            out.append(str(tpl).format(doc_type=doc_type_up, doc_type_lower=doc_type_up.lower()))
        except Exception:
            out.append(str(tpl))
    return out


def load_extractors_for(doc_type: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    routes = _load_routes()
    merged: Dict[str, Any] = {}
    used_rulesets: List[str] = []

    if routes.get("include_common", True):
        common_path = RULES_DIR / "common.json"
        if common_path.exists():
            d = _load_json(common_path)
            merged.update(d.get("extractors") or {})
            used_rulesets.append(common_path.name)

    for name in _resolve_rulesets(doc_type, routes):
        path = RULES_DIR / name
        if not path.exists():
            continue
        d = _load_json(path)
        merged.update(d.get("extractors") or {})
        used_rulesets.append(path.name)

    meta = {"routes_config": routes, "applied_rulesets": used_rulesets}
    return merged, meta


# ----------------- Documents -----------------
def _get_input_docs(globals_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Priorité: selected > TOK_DOCS > FINAL_DOCS > DOCS > TEXT_DOCS
    """
    for key in ("selected", "TOK_DOCS", "FINAL_DOCS", "DOCS", "TEXT_DOCS"):
        val = globals_dict.get(key)
        if isinstance(val, list):
            return val
    raise RuntimeError("Aucune structure de document trouvée (selected/TOK_DOCS/FINAL_DOCS/DOCS/TEXT_DOCS).")


def _page_text_from_page(pg: Dict[str, Any]) -> str:
    if "page_text" in pg:
        return pg.get("page_text") or ""
    if "ocr_text" in pg:
        return pg.get("ocr_text") or ""
    if "sentences_layout" in pg and isinstance(pg.get("sentences_layout"), list):
        parts = []
        for s in pg["sentences_layout"]:
            if isinstance(s, dict):
                parts.append(s.get("text") or "")
            else:
                parts.append(str(s))
        return "\n".join([p for p in parts if p])
    return pg.get("text") or ""


# ----------------- Application des règles -----------------
def apply_extractors_to_page(text: str, extractors: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}

    for name, cfg in extractors.items():
        try:
            regex = _compile_regex(cfg.get("pattern", ""), cfg.get("flags"))
        except re.error as e:
            print(f"[extraction-regles] regex invalide pour '{name}': {e} | pattern={cfg.get('pattern')}")
            continue
        group_idx = int(cfg.get("group", 0) or 0)
        many = bool(cfg.get("many", False))
        surface = (cfg.get("surface") or "text").lower()

        matches: List[Dict[str, Any]] = []

        def _record(m: re.Match, offset: int = 0):
            try:
                val = m.group(group_idx)
            except IndexError:
                val = m.group(0)
            start = (m.start(group_idx) if m.lastindex and group_idx <= m.lastindex else m.start()) + offset
            end = (m.end(group_idx) if m.lastindex and group_idx <= m.lastindex else m.end()) + offset
            matches.append({
                "value": val,
                "start": start,
                "end": end,
                "snippet": text[max(0, start - 40): min(len(text), end + 40)],
            })

        if surface == "lines":
            offset = 0
            for line in text.splitlines(True):
                for m in regex.finditer(line):
                    _record(m, offset)
                offset += len(line)
        else:
            for m in regex.finditer(text):
                _record(m, 0)

        if not many and matches:
            # Conserver uniquement le premier match
            matches = [matches[0]]

        out[name] = matches

    return out


# ----------------- Pipeline principal -----------------
def _build_cls_map(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id = {}
    by_fn = {}
    for r in results or []:
        if r.get("doc_id"):
            by_id[str(r["doc_id"])] = r
        if r.get("filename"):
            by_fn[str(r["filename"])] = r
    return {"by_id": by_id, "by_fn": by_fn}


def _classification_for(doc: Dict[str, Any], cls_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    doc_id = doc.get("doc_id")
    fn = doc.get("filename")
    if doc_id and str(doc_id) in cls_map["by_id"]:
        return cls_map["by_id"][str(doc_id)]
    if fn and str(fn) in cls_map["by_fn"]:
        return cls_map["by_fn"][str(fn)]
    return {"doc_type": "UNCLASSIFIED", "status": "REVIEW"}


def run() -> List[Dict[str, Any]]:
    docs = _get_input_docs(globals())
    results_cls = globals().get("RESULTS") or []
    cls_map = _build_cls_map(results_cls)

    extracted_docs: List[Dict[str, Any]] = []

    for doc in docs:
        cls = _classification_for(doc, cls_map)
        doc_type = (cls.get("doc_type") or "UNCLASSIFIED").upper()
        cls_status = cls.get("status", "REVIEW")

        extractors, meta = load_extractors_for(doc_type)
        if not extractors:
            extracted_docs.append({
                "doc_id": doc.get("doc_id"),
                "filename": doc.get("filename"),
                "doc_type": doc_type,
                "classification_status": cls_status,
                "fields": {},
                "ruleset": meta,
            })
            continue

        fields: Dict[str, Any] = {}

        pages = doc.get("pages") or [{"page_index": 1, "text": doc.get("text", "")}]
        for pg in pages:
            page_text = _page_text_from_page(pg)
            if not page_text:
                continue
            page_res = apply_extractors_to_page(page_text, extractors)
            for name, matches in page_res.items():
                if not matches:
                    continue
                fields.setdefault(name, {
                    "rule_id": extractors[name].get("rule_id"),
                    "type": extractors[name].get("type", "text"),
                    "many": bool(extractors[name].get("many", False)),
                    "matches": [],
                })
                for mt in matches:
                    mt2 = dict(mt)
                    mt2["page_index"] = pg.get("page_index", pg.get("page", 1))
                    fields[name]["matches"].append(mt2)

        for name, cfg in extractors.items():
            fields.setdefault(name, {
                "rule_id": cfg.get("rule_id"),
                "type": cfg.get("type", "text"),
                "many": bool(cfg.get("many", False)),
                "matches": [],
            })

        extracted_doc = {
            "doc_id": doc.get("doc_id"),
            "filename": doc.get("filename"),
            "doc_type": doc_type,
            "classification_status": cls_status,
            "ruleset": meta,
            "fields": fields,
        }

        # Affichage terminal pour debug/traçabilité
        print("\n[extraction]")
        print(f"  doc: {extracted_doc['filename']} | type={doc_type} | status={cls_status}")
        print(f"  rulesets: {', '.join(meta.get('applied_rulesets') or [])}")
        for fname, fcfg in fields.items():
            matches = fcfg.get("matches") or []
            rule_id = fcfg.get("rule_id") or "?"
            print(f"    - {fname} (rule_id={rule_id})  x{len(matches)}")
            for m in matches[:5]:
                val = m.get("value")
                page = m.get("page_index", "?")
                print(f"        p{page}: {val}  [rule_id={rule_id}]")
            if len(matches) > 5:
                print("        ...")

        extracted_docs.append(extracted_doc)

    return extracted_docs


# Exécution directe (runpy.run_path)
EXTRACTIONS = run()
