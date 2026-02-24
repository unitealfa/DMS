# Affiche :
# [classification] <filename> -> best=<DOC_TYPE> | status=<OK/REVIEW> | scores: {...}

import sys, json, re, unicodedata, uuid, ast
from pathlib import Path
from typing import Dict, Any, List, Optional

# ========= CONFIG =========
# Tout est résolu depuis la racine du dépôt pour rester portable
REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATION_DIR = REPO_ROOT / "classification"
COMMON_PATH = CLASSIFICATION_DIR / "common.json"

# Garder le repo root dans sys.path pour des imports éventuels
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ========= HELPERS =========
def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "")
        if unicodedata.category(c) != "Mn"
    )

def _norm_text(s: str) -> str:
    s = _strip_accents(s or "")
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)  # remplace ponctuation par espaces
    return " ".join(s.upper().split())

def _norm_keyword(k: str) -> str:
    """Applique la même normalisation qu'au texte pour garantir le match."""
    return _norm_text(k or "")

def _ensure_kw_dict(d: Dict[str, Any]) -> Dict[str, List[str]]:
    kw = d.get("keywords")
    if isinstance(kw, dict):
        out = {"strong": [], "medium": [], "weak": [], "negative": [], "strong_negative": []}
        for k, v in kw.items():
            if isinstance(v, list):
                kk = str(k).lower()
                if kk in out:
                    out[kk] = [str(x).upper() for x in v]
        return out
    if isinstance(kw, list):
        return {"strong": [], "medium": [], "weak": [str(x).upper() for x in kw], "negative": [], "strong_negative": []}
    if isinstance(d, dict) and all(isinstance(v, list) for v in d.values()):
        flat = []
        for v in d.values():
            flat.extend(v)
        return {"strong": [], "medium": [], "weak": [str(x).upper() for x in flat], "negative": [], "strong_negative": []}
    return {"strong": [], "medium": [], "weak": [], "negative": [], "strong_negative": []}

def load_classification_configs():
    common = {
        "weights": {"strong": 5, "medium": 2, "weak": 1},
        "global_penalties": {"negative": -2, "strong_negative": -5},
        "threshold": 6,
        "margin": 3,
        "tie_breaker": "priority",
    }
    if COMMON_PATH.exists():
        d = _load_json(COMMON_PATH)
        if isinstance(d, dict):
            common.update(d)

    configs = {}
    if CLASSIFICATION_DIR.exists():
        for p in sorted(CLASSIFICATION_DIR.glob("*.json")):
            d = _load_json(p)
            if not isinstance(d, dict):
                continue
            doc_type = str(d.get("doc_type") or p.stem).upper()
            if doc_type == "COMMON":
                continue
            # Skip configs explicitly disabled
            if d.get("enabled") is False:
                continue
            configs[doc_type] = {
                "doc_type": doc_type,
                "keywords": _ensure_kw_dict(d),
                "priority": int(d.get("priority", 0) or 0),
                "anti_confusion_targets": [str(x).upper() for x in d.get("anti_confusion_targets", []) if x],
            }
    return common, configs

def _get_previous_cell_input():
    g = globals()
    for k in ("selected", "TOK_DOCS", "FINAL_DOCS", "DOCS", "TEXT_DOCS", "_"):
        if k in g and g[k] is not None:
            return g[k]
    return None

def _build_DOCS_from_input(data) -> List[Dict[str, Any]]:
    # Cas: list docs avec pages (TOK_DOCS/selected)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "pages" in data[0]:
        out = []
        for i, doc in enumerate(data):
            name = doc.get("filename") or doc.get("doc_id") or f"doc#{i}"
            pages_out = []
            for p_i, pg in enumerate(doc.get("pages") or []):
                page_index = pg.get("page_index", pg.get("page", p_i + 1))
                txt = pg.get("ocr_text")
                if not txt:
                    sent_items = pg.get("sentences_layout") or pg.get("sentences") or pg.get("chunks") or []
                    parts = []
                    for s in sent_items:
                        if isinstance(s, dict):
                            if s.get("is_sentence") is False:
                                continue
                            parts.append(s.get("text") or "")
                        else:
                            parts.append(str(s))
                    txt = "\n".join([x for x in parts if x])
                pages_out.append({"page_index": page_index, "ocr_text": txt or ""})
            out.append({"filename": name, "pages": pages_out})
        return out

    # Cas: FINAL_DOCS list[{text,...}]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
        out = []
        for i, d in enumerate(data):
            name = d.get("filename") or d.get("doc_id") or f"doc#{i}"
            out.append({"filename": name, "pages": [{"page_index": 1, "ocr_text": d.get("text") or ""}]})
        return out

    # Cas: dict {text:...}
    if isinstance(data, dict) and "text" in data:
        name = data.get("filename") or data.get("doc_id") or "doc"
        return [{"filename": name, "pages": [{"page_index": 1, "ocr_text": data.get("text") or ""}]}]

    # Cas: string
    if isinstance(data, str):
        return [{"filename": "text", "pages": [{"page_index": 1, "ocr_text": data}]}]

    raise TypeError(f"Format d'entrée non supporté: {type(data)}")

def classify_scores(DOCS: List[Dict[str, Any]], common: Dict[str, Any], configs: Dict[str, Any]) -> None:
    weights = common.get("weights", {"strong": 5, "medium": 2, "weak": 1})
    penalties = common.get("global_penalties", {"negative": -2, "strong_negative": -5})

    for doc in DOCS:
        scores_doc = {dt: 0 for dt in configs.keys()}
        matches_doc = {
            dt: {"strong": {}, "medium": {}, "weak": {}, "negative": {}, "strong_negative": {}, "anti_confusion": {}}
            for dt in configs.keys()
        }

        def add_score(text: str, keywords: List[str], delta: int, bucket: str, dt: str, weight_factor: float) -> int:
            if not keywords:
                return 0
            s = 0
            for k in keywords:
                k_norm = _norm_keyword(k)
                if not k_norm:
                    continue
                # mot/phrase entier(e) avec espaces tolérant la ponctuation (déjà normalisée en spaces)
                pattern = r"\b" + re.escape(k_norm).replace(r"\ ", r"\s+") + r"\b"
                occ = len(re.findall(pattern, text))
                if occ:
                    gain = int(delta * occ * weight_factor)
                    s += gain
                    matches_doc[dt][bucket][k_norm] = matches_doc[dt][bucket].get(k_norm, 0) + occ
            return s

        for page in doc.get("pages", []):
            page_idx = page.get("page_index", page.get("page", 1)) or 1
            weight_factor = 1.2 if page_idx == 1 else 1.0  # la page 1 pèse un peu plus
            text = _norm_text(page.get("ocr_text", ""))
            for dt, cfg in configs.items():
                kw = cfg["keywords"]
                score = 0
                score += add_score(text, kw.get("strong", []), int(weights.get("strong", 5)), "strong", dt, weight_factor)
                score += add_score(text, kw.get("medium", []), int(weights.get("medium", 2)), "medium", dt, weight_factor)
                score += add_score(text, kw.get("weak", []), int(weights.get("weak", 1)), "weak", dt, weight_factor)
                score += add_score(text, kw.get("negative", []), int(penalties.get("negative", -2)), "negative", dt, weight_factor)
                score += add_score(text, kw.get("strong_negative", []), int(penalties.get("strong_negative", -5)), "strong_negative", dt, weight_factor)

                # Anti-confusion : appliquer un malus si des termes concurrents sont présents
                for tgt in cfg.get("anti_confusion_targets", []):
                    tgt_norm = _norm_keyword(tgt)
                    if not tgt_norm:
                        continue
                    # si c'est un doc_type style BON_DE_COMMANDE (pas dans le texte), on ignore
                    if "_" in tgt_norm:
                        continue
                    pattern_c = r"\b" + re.escape(tgt_norm).replace(r"\ ", r"\s+") + r"\b"
                    occ_c = len(re.findall(pattern_c, text))
                    if occ_c:
                        malus = int(weights.get("strong", 5)) * occ_c * weight_factor
                        score -= malus
                        matches_doc[dt]["anti_confusion"][tgt_norm] = matches_doc[dt]["anti_confusion"].get(tgt_norm, 0) + occ_c

                scores_doc[dt] += score
        doc["scores"] = scores_doc
        doc["score_matches"] = matches_doc

# ========= DECISION =========
def decide(scores: Dict[str, int], configs: Dict[str, Any], common: Dict[str, Any]) -> Dict[str, Any]:
    THRESHOLD = int(common.get("threshold", 6))
    MARGIN = int(common.get("margin", 3))

    PRIORITY = {dt: int((configs.get(dt, {}) or {}).get("priority", 0) or 0) for dt in configs.keys()}
    scores_stable = {dt: int(scores.get(dt, 0)) for dt in configs.keys()}

    # tri stable: score desc, priority desc, name asc
    items = sorted(
        scores_stable.items(),
        key=lambda kv: (-kv[1], -PRIORITY.get(kv[0], 0), kv[0])
    )
    top_type, top_score = items[0] if items else ("UNCLASSIFIED", 0)
    second_score = items[1][1] if len(items) > 1 else 0
    diff = top_score - second_score

    confident = (top_score > 0) and (top_score >= THRESHOLD) and (diff >= MARGIN)
    best = top_type if confident else "UNCLASSIFIED"
    status = "OK" if confident else "REVIEW"

    return {
        "best": best,
        "status": status,
        "top_score": top_score,
        "second_score": second_score,
        "diff": diff,
        "scores_stable": scores_stable
    }

# ========= RUN =========
data = _get_previous_cell_input()
if data is None:
    raise RuntimeError("Je ne trouve pas de données d'entrée (selected / TOK_DOCS / FINAL_DOCS / DOCS / TEXT_DOCS).")

common, configs = load_classification_configs()
if not configs:
    raise RuntimeError(f"Aucune classe trouvée dans: {CLASSIFICATION_DIR}")

DOCS = _build_DOCS_from_input(data)
classify_scores(DOCS, common, configs)

preferred = ["ARTICLE", "BON_DE_COMMANDE", "CONTRAT", "FACTURE", "FORMULAIRE"]
order = [c for c in preferred if c in configs] + [c for c in configs.keys() if c not in preferred]

RESULTS = []
for doc in DOCS:
    scores = doc.get("scores", {}) or {}
    ordered_scores = {k: int(scores.get(k, 0)) for k in order}

    d = decide(scores, configs, common)
    best, status = d["best"], d["status"]

    # ---- sortie lisible + compacte (tu vois enfin la classe)
    print(f"[classification] {doc.get('filename')} -> best={best} | status={status} | scores: {ordered_scores}")

    # Détails: mots-clés matchés et cibles d'anti-confusion pour la classe retenue
    if best in configs:
        matches = (doc.get("score_matches") or {}).get(best, {})
        ac_targets = configs[best].get("anti_confusion_targets", [])
        def _fmt(bucket: str) -> str:
            vals = matches.get(bucket) or {}
            if not vals:
                return f"{bucket}=[]"
            items = [f"{k}(x{v})" for k, v in vals.items()]
            return f"{bucket}=[" + ", ".join(items) + "]"
        print("  keywords:", _fmt("strong"), _fmt("medium"), _fmt("weak"), _fmt("negative"))
        if matches.get("strong_negative"):
            print("  keywords strong_negative:", ", ".join(f"{k}(x{v})" for k,v in matches["strong_negative"].items()))
        if matches.get("anti_confusion"):
            print("  anti_confusion hits:", ", ".join(f"{k}(x{v})" for k,v in matches["anti_confusion"].items()))
        if ac_targets:
            print("  anti_confusion_targets:", ", ".join(ac_targets))

    # (optionnel) stocker le json détaillé sans l'imprimer
    doc["result"] = {
        "doc_id": doc.get("doc_id") or str(uuid.uuid4()),
        "filename": doc.get("filename"),
        "doc_type": best,
        "status": status,
        "scores": d["scores_stable"],
        "threshold": int(common.get("threshold", 6)),
        "margin": int(common.get("margin", 3)),
        "decision_debug": {
            "top_score": d["top_score"],
            "second_score": d["second_score"],
            "diff": d["diff"],
        }
    }
    RESULTS.append(doc["result"])
