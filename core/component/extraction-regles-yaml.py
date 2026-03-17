"""
Extraction des champs metier via regles YAML (sans regex de champs).
- Selection des regles par doc_type (classification) via config/ruleset_routes.yaml.
- Chargement des extracteurs YAML dans rules/*.yaml.
- Extraction par labels (line-based) + detecteurs simples (date/email/phone/url/currency/amount).
- Conserve un format de sortie compatible avec EXTRACTIONS du pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - fallback environnement sans PyYAML
    yaml = None

# Dossiers
REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_DIR = REPO_ROOT / "rules"
CONFIG_DIR = REPO_ROOT / "config"
ROUTES_YAML_PATH = CONFIG_DIR / "ruleset_routes.yaml"
ROUTES_JSON_FALLBACK = CONFIG_DIR / "ruleset_routes.json"

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
SEPARATORS = [":", "=", "-", "–", "—", "؛"]
AMOUNT_ALLOWED = set("0123456789٠١٢٣٤٥٦٧٨٩.,'٬٫ +-")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"]+", re.IGNORECASE)

_MONTH_TOKENS = [
    "janvier",
    "fevrier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
    "décembre",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_WEEKDAY_TOKENS = [
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
_MONTH_PATTERN = "|".join(sorted((re.escape(m) for m in _MONTH_TOKENS), key=len, reverse=True))
_WEEKDAY_PATTERN = "|".join(sorted((re.escape(d) for d in _WEEKDAY_TOKENS), key=len, reverse=True))
DATE_WORD_DMY_RE = re.compile(
    rf"\b(?:(?:{_WEEKDAY_PATTERN})\s+)?\d{{1,2}}(?:er)?\s+(?:de\s+)?(?:{_MONTH_PATTERN})\s+\d{{2,4}}\b",
    re.IGNORECASE,
)
DATE_WORD_MDY_RE = re.compile(
    rf"\b(?:{_MONTH_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{2,4}}\b",
    re.IGNORECASE,
)

_CURRENCY_ALIASES: List[Tuple[str, str]] = [
    ("eur", "EUR"),
    ("euro", "EUR"),
    ("euros", "EUR"),
    ("€", "EUR"),
    ("usd", "USD"),
    ("dollar", "USD"),
    ("dollars", "USD"),
    ("$", "USD"),
    ("gbp", "GBP"),
    ("pound", "GBP"),
    ("£", "GBP"),
    ("dzd", "DZD"),
    ("da", "DZD"),
    ("d.a", "DZD"),
    ("دج", "DZD"),
    ("د.ج", "DZD"),
    ("دينار", "DZD"),
    ("دينار جزائري", "DZD"),
    ("aed", "AED"),
    ("sar", "SAR"),
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_json_like_yaml(path: Path) -> Dict[str, Any]:
    # JSON est un sous-ensemble YAML: permet de lire aussi l'ancien fichier .json si necessaire.
    return _load_yaml(path)


def _load_routes() -> Dict[str, Any]:
    default = {
        "include_common": True,
        "default_rulesets": ["{doc_type}.yaml"],
        "ruleset_mapping": {},
    }
    cfg: Dict[str, Any] = {}
    if ROUTES_YAML_PATH.exists():
        cfg = _load_yaml(ROUTES_YAML_PATH)
    elif ROUTES_JSON_FALLBACK.exists():
        cfg = _load_json_like_yaml(ROUTES_JSON_FALLBACK)

    if not isinstance(cfg, dict):
        return default
    cfg.setdefault("include_common", True)
    cfg.setdefault("default_rulesets", ["{doc_type}.yaml"])
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
        common_path = RULES_DIR / "common.yaml"
        if common_path.exists():
            d = _load_yaml(common_path)
            merged.update(d.get("extractors") or {})
            used_rulesets.append(common_path.name)

    for name in _resolve_rulesets(doc_type, routes):
        path = RULES_DIR / str(name)
        if not path.exists():
            continue
        d = _load_yaml(path)
        merged.update(d.get("extractors") or {})
        used_rulesets.append(path.name)

    meta = {"routes_config": routes, "applied_rulesets": used_rulesets}
    return merged, meta


def _get_input_docs(globals_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Priorite identique au composant regex.
    for key in ("ES_EXTRACTION_DOCS", "selected", "TOK_DOCS", "FINAL_DOCS", "DOCS", "TEXT_DOCS"):
        val = globals_dict.get(key)
        if isinstance(val, list) and val:
            return val
    raise RuntimeError("Aucune structure de document trouvée (selected/TOK_DOCS/FINAL_DOCS/DOCS/TEXT_DOCS).")


def _page_text_from_page(pg: Dict[str, Any]) -> str:
    if "page_text" in pg:
        return pg.get("page_text") or ""
    if "ocr_text" in pg:
        return pg.get("ocr_text") or ""
    if "sentences_layout" in pg and isinstance(pg.get("sentences_layout"), list):
        parts: List[str] = []
        for s in pg["sentences_layout"]:
            if isinstance(s, dict):
                parts.append(s.get("text") or "")
            else:
                parts.append(str(s))
        return "\n".join([p for p in parts if p])
    return pg.get("text") or ""


def _norm_text(s: Any) -> str:
    text = str(s or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().split())


def _normalize_digits(s: str) -> str:
    return str(s or "").translate(ARABIC_DIGITS)


def _strip_control_and_escapes(value: str) -> str:
    text = str(value or "")
    for old, new in (
        ("\\r\\n", " "),
        ("\\n", " "),
        ("\\r", " "),
        ("\\t", " "),
        ("\\xa0", " "),
        ("\\u00a0", " "),
    ):
        text = text.replace(old, new)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("\xa0", " ")
    text = "".join(ch if (ch.isprintable() or ch.isspace()) else " " for ch in text)
    return " ".join(text.split())


def _clean_value(value: str) -> str:
    val = _strip_control_and_escapes(value).strip(" \t\r\n:=-–—;,.\"'`")
    if len(val) >= 2 and ((val[0] == "'" and val[-1] == "'") or (val[0] == '"' and val[-1] == '"')):
        val = val[1:-1].strip(" \t\r\n:=-–—;,.\"'`")
    return val


def _normalize_email_value(raw_value: str) -> str:
    text = _strip_control_and_escapes(raw_value)
    m = EMAIL_RE.search(text)
    if not m:
        return ""
    val = str(m.group(0) or "").strip(" \t\r\n<>()[]{}\"'`,;")
    val = val.rstrip(".")
    return val.lower()


def _normalize_url_value(raw_value: str) -> str:
    text = _strip_control_and_escapes(raw_value)
    m = URL_RE.search(text)
    if not m:
        return ""
    val = str(m.group(0) or "").strip(" \t\r\n<>()[]{}\"'`,;")
    return val.rstrip(".")


def _normalize_phone_value(raw_value: str) -> str:
    text = _normalize_digits(_strip_control_and_escapes(raw_value))
    if not text:
        return ""
    has_plus = "+" in text
    digits = "".join(ch for ch in text if ch.isdigit())
    if not (10 <= len(digits) <= 15):
        return ""
    return f"+{digits}" if has_plus else digits


def _first_non_empty_line(lines: List[str], start: int, max_lookahead: int) -> str:
    upper = min(len(lines), start + max(1, max_lookahead) + 1)
    for i in range(start, upper):
        val = _clean_value(lines[i])
        if val:
            return val
    return ""


def _find_label_in_line(line: str, labels: List[str]) -> Tuple[int, str]:
    low = line.lower()
    for label in labels:
        lab = str(label or "").strip()
        if not lab:
            continue
        idx = low.find(lab.lower())
        if idx >= 0:
            return idx, lab

    line_norm = _norm_text(line)
    for label in labels:
        lab = str(label or "").strip()
        if not lab:
            continue
        if _norm_text(lab) in line_norm:
            return -1, lab
    return -1, ""


def _value_after_label(line: str, label: str, label_idx: int) -> str:
    if label_idx >= 0:
        start = max(0, label_idx)
        sep_pos = -1
        for sep in SEPARATORS:
            pos = line.find(sep, start)
            if pos >= 0 and (sep_pos < 0 or pos < sep_pos):
                sep_pos = pos
        if sep_pos >= 0:
            val = _clean_value(line[sep_pos + 1 :])
            if val:
                return val

        low = line.lower()
        lab_low = label.lower()
        p = low.find(lab_low)
        if p >= 0:
            val = _clean_value(line[p + len(label) :])
            if val:
                return val

    # Fallback: ligne entiere
    return _clean_value(line)


def _split_tokens_rough(text: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    for ch in str(text or ""):
        if ch.isspace():
            if buf:
                out.append("".join(buf))
                buf = []
            continue
        if ch in ",;()[]{}<>":
            if buf:
                out.append("".join(buf))
                buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _extract_date_value(text: str) -> str:
    source = _normalize_digits(_strip_control_and_escapes(text))
    for tok in _split_tokens_rough(source):
        t = tok.strip(" .,:;")
        for sep in ("/", "-", "."):
            if t.count(sep) != 2:
                continue
            parts = [x for x in t.split(sep) if x]
            if len(parts) != 3 or not all(p.isdigit() for p in parts):
                continue
            # Formats dd/mm/yyyy ou yyyy/mm/dd
            if len(parts[2]) == 4 and 1 <= int(parts[0]) <= 31 and 1 <= int(parts[1]) <= 12:
                return f"{int(parts[0]):02d}/{int(parts[1]):02d}/{parts[2]}"
            if len(parts[0]) == 4 and 1 <= int(parts[1]) <= 12 and 1 <= int(parts[2]) <= 31:
                return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    for rx in (DATE_WORD_DMY_RE, DATE_WORD_MDY_RE):
        m = rx.search(source)
        if m:
            return _clean_value(m.group(0))
    return ""


def _extract_amount_value(text: str) -> str:
    best = ""
    for tok in _split_tokens_rough(text):
        raw = "".join(ch for ch in tok if ch in AMOUNT_ALLOWED)
        cand = _normalize_digits(raw).strip()
        digit_count = sum(ch.isdigit() for ch in cand)
        if digit_count < 2:
            continue
        if digit_count > sum(ch.isdigit() for ch in best):
            best = cand
    return _clean_value(best)


def _extract_currency_value(text: str) -> str:
    low = _norm_text(text)
    for alias, code in _CURRENCY_ALIASES:
        if _norm_text(alias) in low:
            return code
    return ""


def _detect_emails(text: str) -> List[str]:
    out: List[str] = []
    src = _strip_control_and_escapes(text)
    for m in EMAIL_RE.finditer(src):
        val = _normalize_email_value(m.group(0))
        if val:
            out.append(val)
    if not out:
        val = _normalize_email_value(src)
        if val:
            out.append(val)
    return _unique_keep_order(out)


def _detect_urls(text: str) -> List[str]:
    out: List[str] = []
    src = _strip_control_and_escapes(text)
    for m in URL_RE.finditer(src):
        val = _normalize_url_value(m.group(0))
        if val:
            out.append(val)
    return _unique_keep_order(out)


def _detect_phones(text: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    allowed = set("0123456789٠١٢٣٤٥٦٧٨٩+ .-()/")
    src = _normalize_digits(_strip_control_and_escapes(text))
    for ch in src + " ":
        if ch in allowed:
            buf.append(ch)
            continue
        if buf:
            chunk = "".join(buf).strip()
            buf = []
            val = _normalize_phone_value(chunk)
            if val:
                out.append(val)
    return _unique_keep_order(out)


def _detect_values_by_type(value_type: str, text: str) -> List[str]:
    t = (value_type or "text").lower()
    if t == "email":
        return _detect_emails(text)
    if t in {"url", "site"}:
        return _detect_urls(text)
    if t in {"phone", "telephone"}:
        return _detect_phones(text)
    if t == "date":
        val = _extract_date_value(text)
        return [val] if val else []
    if t in {"amount", "money"}:
        val = _extract_amount_value(text)
        return [val] if val else []
    if t == "currency":
        val = _extract_currency_value(text)
        return [val] if val else []
    return []


def _normalize_value_by_type(value_type: str, raw_value: str) -> str:
    t = (value_type or "text").lower()
    v = _clean_value(raw_value)
    if not v:
        return ""
    if t == "email":
        return _normalize_email_value(v) or ""
    if t in {"url", "site"}:
        return _normalize_url_value(v) or ""
    if t in {"phone", "telephone"}:
        return _normalize_phone_value(v) or ""
    if t == "date":
        return _extract_date_value(v) or ""
    if t in {"amount", "money"}:
        return _extract_amount_value(v) or ""
    if t == "currency":
        return _extract_currency_value(v) or ""
    return v


def _make_match(page_text: str, value: str, page_index: Any, fallback_snippet: str) -> Dict[str, Any]:
    start = page_text.find(value) if value else -1
    if start >= 0:
        end = start + len(value)
        snippet = page_text[max(0, start - 40) : min(len(page_text), end + 40)]
    else:
        end = -1
        snippet = fallback_snippet[:320]
    return {
        "value": value,
        "start": int(start),
        "end": int(end),
        "snippet": snippet,
        "page_index": page_index,
    }


def _apply_extractor_to_page(page_text: str, page_index: Any, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(cfg, dict):
        return []

    value_type = str(cfg.get("type") or "text")
    many = bool(cfg.get("many", False))
    strategy = str(cfg.get("strategy") or "label_value").strip().lower()
    labels = [str(x) for x in (cfg.get("labels") or []) if str(x).strip()]
    lookahead = int(cfg.get("lookahead_lines") or 1)
    min_chars = int(cfg.get("min_chars") or 1)
    max_chars = int(cfg.get("max_chars") or 500)

    matches: List[Dict[str, Any]] = []

    if strategy == "detector":
        vals = _detect_values_by_type(value_type, page_text)
        for val in vals:
            if not val or len(val) < min_chars or len(val) > max_chars:
                continue
            matches.append(_make_match(page_text, val, page_index, val))
            if not many:
                break
        return matches

    # Strategie line-based par labels.
    lines = page_text.splitlines()
    for i, line in enumerate(lines):
        idx, lab = _find_label_in_line(line, labels)
        if not lab:
            continue

        value = _value_after_label(line, lab, idx)
        if not value:
            value = _first_non_empty_line(lines, i + 1, lookahead)

        norm_value = _normalize_value_by_type(value_type, value)
        if not norm_value:
            continue
        if len(norm_value) < min_chars or len(norm_value) > max_chars:
            continue

        matches.append(_make_match(page_text, norm_value, page_index, line))
        if not many:
            break

    # Fallback detecteur pour types techniques si aucune valeur trouvee par labels.
    if not matches and value_type.lower() in {"email", "url", "site", "phone", "telephone", "currency"}:
        for val in _detect_values_by_type(value_type, page_text):
            if len(val) < min_chars or len(val) > max_chars:
                continue
            matches.append(_make_match(page_text, val, page_index, val))
            if not many:
                break

    return matches


def _build_cls_map(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_fn: Dict[str, Dict[str, Any]] = {}
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


def _doc_text_score(doc: Dict[str, Any]) -> int:
    pages = doc.get("pages")
    if isinstance(pages, list) and pages:
        total = 0
        for pg in pages:
            if not isinstance(pg, dict):
                continue
            total += len(str(_page_text_from_page(pg)).strip())
        if total > 0:
            return total
    return len(str(doc.get("text") or "").strip())


def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        key = str(doc.get("doc_id") or "").strip() or str(doc.get("filename") or "").strip() or f"doc#{i}"
        if key not in best:
            best[key] = doc
            order.append(key)
            continue
        cur = best[key]
        cur_score = _doc_text_score(cur)
        new_score = _doc_text_score(doc)
        cur_content = str(cur.get("content") or "").strip().lower()
        new_content = str(doc.get("content") or "").strip().lower()
        replace = False
        if cur_content == "image_only" and new_content != "image_only":
            replace = True
        elif cur_content != "image_only" and new_content == "image_only":
            replace = False
        elif new_score > cur_score:
            replace = True
        if replace:
            best[key] = doc
    return [best[k] for k in order]


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def run() -> List[Dict[str, Any]]:
    docs = _dedupe_docs(_get_input_docs(globals()))
    results_cls = globals().get("RESULTS") or []
    cls_map = _build_cls_map(results_cls)

    extracted_docs: List[Dict[str, Any]] = []

    for doc in docs:
        cls = _classification_for(doc, cls_map)
        doc_type = (cls.get("doc_type") or "UNCLASSIFIED").upper()
        cls_status = cls.get("status", "REVIEW")

        extractors, meta = load_extractors_for(doc_type)
        if not extractors:
            extracted_docs.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "filename": doc.get("filename"),
                    "doc_type": doc_type,
                    "classification_status": cls_status,
                    "fields": {},
                    "ruleset": meta,
                }
            )
            continue

        fields: Dict[str, Any] = {}
        pages = doc.get("pages") or [{"page_index": 1, "text": doc.get("text", "")}]

        for pg in pages:
            page_text = _page_text_from_page(pg)
            if not page_text:
                continue
            page_index = pg.get("page_index", pg.get("page", 1))

            for name, cfg in extractors.items():
                if not isinstance(cfg, dict):
                    continue
                fields.setdefault(
                    name,
                    {
                        "rule_id": cfg.get("rule_id"),
                        "type": cfg.get("type", "text"),
                        "many": bool(cfg.get("many", False)),
                        "matches": [],
                    },
                )
                page_matches = _apply_extractor_to_page(page_text, page_index, cfg)
                if page_matches:
                    fields[name]["matches"].extend(page_matches)

        # Dedup des valeurs par champ
        for name, field in fields.items():
            raw_matches = field.get("matches") or []
            unique_vals = _unique_keep_order([str(m.get("value") or "") for m in raw_matches if m.get("value")])
            dedup_matches: List[Dict[str, Any]] = []
            for val in unique_vals:
                for m in raw_matches:
                    if str(m.get("value") or "") == val:
                        dedup_matches.append(m)
                        break
            if not bool(field.get("many", False)) and dedup_matches:
                dedup_matches = [dedup_matches[0]]
            field["matches"] = dedup_matches

        extracted_doc = {
            "doc_id": doc.get("doc_id"),
            "filename": doc.get("filename"),
            "doc_type": doc_type,
            "classification_status": cls_status,
            "ruleset": meta,
            "fields": fields,
        }

        print("\n[extraction-yaml]")
        print(f"  doc: {extracted_doc['filename']} | type={doc_type} | status={cls_status}")
        print(f"  rulesets: {', '.join(meta.get('applied_rulesets') or [])}")
        for fname, fcfg in fields.items():
            matches = fcfg.get("matches") or []
            rule_id = fcfg.get("rule_id") or "?"
            print(f"    - {fname} (rule_id={rule_id})  x{len(matches)}")
            for m in matches:
                val = m.get("value")
                page = m.get("page_index", "?")
                print(f"        p{page}: {val}  [rule_id={rule_id}]")

        extracted_docs.append(extracted_doc)

    return extracted_docs


# Execution directe (runpy.run_path)
EXTRACTIONS = run()
