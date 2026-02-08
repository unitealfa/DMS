# build_resources.py (OFFLINE-ONLY)
from __future__ import annotations

import re
import csv
import json
import hashlib
from pathlib import Path

# ----------------------------
# ROOT PATH (lanceable depuis n'importe où)
# ----------------------------
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "resources"
CACHE_DIR = OUT_DIR / "_cache"

# ----------------------------
# Filtrage / normalisation
# ----------------------------
MIN_LEN = 2
DROP_ALL_DIGITS = True
DROP_MOSTLY_PUNCT = True
_PUNCT_HEAVY = re.compile(r"^[\W_]+$", flags=re.UNICODE)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def normalize_entity(s: str) -> str | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("“”\"'`[](){}<>")
    if len(s) < MIN_LEN:
        return None
    if DROP_ALL_DIGITS and re.fullmatch(r"\d+", s):
        return None
    if DROP_MOSTLY_PUNCT and _PUNCT_HEAVY.match(s):
        return None
    return s

def uniq_sorted(items: list[str]) -> list[str]:
    seen = {}
    for x in items:
        if x not in seen:
            seen[x] = None
    return sorted(seen.keys(), key=lambda t: (t.casefold(), t))

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_lines(path: Path, items: list[str]):
    ensure_dir(path.parent)
    path.write_text("\n".join(items) + "\n", encoding="utf-8")

def iter_strings(obj):
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for x in obj:
            yield from iter_strings(x)

# ----------------------------
# Vérification cache (100% local)
# ----------------------------
def require_exists(p: Path, hint: str):
    if not p.exists():
        raise RuntimeError(f"MANQUE: {p}\nAttendu: {hint}")

def prepare_sources_offline():
    require_exists(OUT_DIR, "Dossier resources/ à la racine du projet")
    require_exists(CACHE_DIR, "Dossier resources/_cache/")

    corpora_dir = CACHE_DIR / "corpora"
    wordlists_dir = CACHE_DIR / "wordlists"
    cscdb_dir = CACHE_DIR / "countries-states-cities-database"

    require_exists(corpora_dir / "data", r"resources\_cache\corpora\data\...")
    require_exists(wordlists_dir, r"resources\_cache\wordlists\...")
    require_exists(cscdb_dir / "csv" / "cities.csv", r"resources\_cache\countries-states-cities-database\csv\cities.csv")

    countries_path = CACHE_DIR / "countries.json"
    en_lemma_path = CACHE_DIR / "en_lemma_lookup.json"
    fr_lemma_path = CACHE_DIR / "fr_lemma_lookup.json"

    require_exists(countries_path, r"resources\_cache\countries.json")
    require_exists(en_lemma_path, r"resources\_cache\en_lemma_lookup.json")
    require_exists(fr_lemma_path, r"resources\_cache\fr_lemma_lookup.json")

    return {
        "corpora_dir": corpora_dir,
        "wordlists_dir": wordlists_dir,
        "cscdb_dir": cscdb_dir,
        "countries_json": countries_path,
        "en_lemma_lookup": en_lemma_path,
        "fr_lemma_lookup": fr_lemma_path,
        "wikidata_cache_dir": CACHE_DIR / "wikidata",  # optionnel
    }

# ----------------------------
# Gazetteers: PER (local)
# ----------------------------
def build_per_lists(paths):
    corpora = paths["corpora_dir"]
    wordlists = paths["wordlists_dir"]

    out = {"en": [], "fr": [], "ar": []}

    # corpora humans (EN)
    humans_dir = corpora / "data" / "humans"
    for fn in ["firstNames.json", "lastNames.json", "names.json"]:
        p = humans_dir / fn
        if p.exists():
            j = read_json(p)
            for s in iter_strings(j):
                ns = normalize_entity(s)
                if ns:
                    out["en"].append(ns)

    # wordlists surnames (EN/FR)
    def load_txt(p: Path) -> list[str]:
        if not p.exists():
            return []
        res = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            n = normalize_entity(line)
            if n:
                res.append(n)
                t = n.title()
                if t != n:
                    res.append(t)
        return res

    out["en"] += load_txt(wordlists / "names" / "surnames" / "english.txt")
    out["fr"] += load_txt(wordlists / "names" / "surnames" / "french.txt")
    # pas d'arabic.txt dans ton repo wordlists -> on ignore

    # Bonus local: wordlists/names/people/*.txt (EN) => PER (noms/people lists)
    people_dir = wordlists / "names" / "people"
    if people_dir.exists():
        for p in people_dir.glob("*.txt"):
            out["en"] += load_txt(p)

    for lang in out:
        out[lang] = uniq_sorted([x for x in (normalize_entity(y) for y in out[lang]) if x])

    return out

# ----------------------------
# Gazetteers: LOC (local)
# ----------------------------
def build_loc_lists(paths):
    countries = json.loads(paths["countries_json"].read_text(encoding="utf-8"))
    cscdb = paths["cscdb_dir"]

    out = {"en": [], "fr": [], "ar": []}

    lang_map = {"en": "eng", "fr": "fra", "ar": "ara"}
    for c in countries:
        name = c.get("name", {}) if isinstance(c.get("name"), dict) else {}
        for k in ["common", "official"]:
            v = name.get(k)
            nv = normalize_entity(v) if isinstance(v, str) else None
            if nv:
                out["en"].append(nv)

        tr = c.get("translations", {}) if isinstance(c.get("translations"), dict) else {}
        for lang, code3 in lang_map.items():
            tblock = tr.get(code3)
            if isinstance(tblock, dict):
                for k in ["common", "official"]:
                    v = tblock.get(k)
                    nv = normalize_entity(v) if isinstance(v, str) else None
                    if nv:
                        out[lang].append(nv)

        cap = c.get("capital")
        if isinstance(cap, list):
            for x in cap:
                nx = normalize_entity(x) if isinstance(x, str) else None
                if nx:
                    out["en"].append(nx)
                    out["fr"].append(nx)
                    out["ar"].append(nx)

    # Cities (CSV local)
    cities_csv = cscdb / "csv" / "cities.csv"
    with cities_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name") or ""
            n = normalize_entity(name)
            if n:
                out["en"].append(n)
                out["fr"].append(n)
                out["ar"].append(n)

    for lang in out:
        out[lang] = uniq_sorted([x for x in (normalize_entity(y) for y in out[lang]) if x])

    return out

# ----------------------------
# Gazetteers: ORG (local)
# ----------------------------
LEGAL_FORMS = {
    "en": ["LLC", "LTD", "INC", "CORP", "PLC", "GMBH", "S.A.", "SAS"],
    "fr": ["SARL", "EURL", "SAS", "SASU", "SA", "SNC", "SCOP", "ASSOCIATION"],
    "ar": ["شركة", "مؤسسة", "جمعية"],
}

def build_org_lists(paths):
    corpora = paths["corpora_dir"]
    out = {"en": [], "fr": [], "ar": []}

    corp_dir = corpora / "data" / "corporations"
    if corp_dir.exists():
        for p in corp_dir.glob("*.json"):
            try:
                j = read_json(p)
                for s in iter_strings(j):
                    ns = normalize_entity(s)
                    if ns:
                        out["en"].append(ns)
            except Exception:
                pass

    for lang, forms in LEGAL_FORMS.items():
        for f in forms:
            nf = normalize_entity(f)
            if nf:
                out[lang].append(nf)

    for lang in out:
        out[lang] = uniq_sorted([x for x in (normalize_entity(y) for y in out[lang]) if x])

    return out

# ----------------------------
# Wikidata (offline cache optionnel)
# ----------------------------
def apply_wikidata_cache_if_present(paths, per, org, loc):
    base = paths["wikidata_cache_dir"]
    if not base.exists():
        return  # pas de cache -> on ignore

    for lang in ["en", "fr", "ar"]:
        for typ, target in [("PER", per), ("ORG", org), ("LOC", loc)]:
            p = base / lang / f"{typ}.txt"
            if not p.exists():
                continue
            extra = [normalize_entity(x) for x in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
            extra = [x for x in extra if x]
            target[lang] = uniq_sorted(target[lang] + extra)

# ----------------------------
# Lemmas -> TSV (local)
# ----------------------------
def build_lemma_tsv(paths):
    en_map = json.loads(paths["en_lemma_lookup"].read_text(encoding="utf-8"))
    fr_map = json.loads(paths["fr_lemma_lookup"].read_text(encoding="utf-8"))

    def dump_tsv(m: dict[str, str], out_path: Path):
        ensure_dir(out_path.parent)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            for k in sorted(m.keys(), key=lambda t: (t.casefold(), t)):
                v = m[k]
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                kk = k.strip()
                vv = v.strip()
                if not kk or not vv:
                    continue
                f.write(f"{kk}\t{vv}\n")

    dump_tsv(en_map, OUT_DIR / "lemmas" / "en.tsv")
    dump_tsv(fr_map, OUT_DIR / "lemmas" / "fr.tsv")

# ----------------------------
# POS verbs (local)
# ----------------------------
def build_verbs(paths):
    corpora = paths["corpora_dir"]
    verbs_en = set()
    verbs_fr = set()

    vpath = corpora / "data" / "words" / "verbs.json"
    if vpath.exists():
        j = read_json(vpath)
        for s in iter_strings(j):
            ns = normalize_entity(s)
            if ns:
                verbs_en.add(ns.lower())

    write_lines(OUT_DIR / "pos" / "en_verbs.txt", uniq_sorted(list(verbs_en)))

    fr_map = json.loads(paths["fr_lemma_lookup"].read_text(encoding="utf-8"))
    rx = re.compile(r".+(er|ir|re|oir)$", flags=re.IGNORECASE)
    for lemma in fr_map.values():
        if not isinstance(lemma, str):
            continue
        lemma = lemma.strip().lower()
        if lemma and rx.match(lemma):
            verbs_fr.add(lemma)

    write_lines(OUT_DIR / "pos" / "fr_verbs.txt", uniq_sorted(list(verbs_fr)))

# ----------------------------
# Write outputs
# ----------------------------
def write_gazetteers(per, org, loc):
    for lang in ["en", "fr", "ar"]:
        write_lines(OUT_DIR / "gazetteers" / lang / "PER.txt", per[lang])
        write_lines(OUT_DIR / "gazetteers" / lang / "ORG.txt", org[lang])
        write_lines(OUT_DIR / "gazetteers" / lang / "LOC.txt", loc[lang])

def make_manifest(paths, per, org, loc):
    man = {
        "offline": True,
        "cache_dir": str(CACHE_DIR),
        "generated": {
            "gazetteers": {
                "en": {"PER": len(per["en"]), "ORG": len(org["en"]), "LOC": len(loc["en"])},
                "fr": {"PER": len(per["fr"]), "ORG": len(org["fr"]), "LOC": len(loc["fr"])},
                "ar": {"PER": len(per["ar"]), "ORG": len(org["ar"]), "LOC": len(loc["ar"])},
            }
        },
        "wikidata_cache_used": (paths["wikidata_cache_dir"].exists()),
    }
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    paths = prepare_sources_offline()

    per = build_per_lists(paths)
    loc = build_loc_lists(paths)
    org = build_org_lists(paths)

    # Optionnel: merge cache wikidata si présent (sans internet)
    apply_wikidata_cache_if_present(paths, per, org, loc)

    write_gazetteers(per, org, loc)
    build_lemma_tsv(paths)
    build_verbs(paths)
    make_manifest(paths, per, org, loc)

    print("OK (offline). Outputs générés dans:", OUT_DIR)
    print("  - resources/gazetteers/<lang>/{PER,ORG,LOC}.txt")
    print("  - resources/pos/{en_verbs,fr_verbs}.txt")
    print("  - resources/lemmas/{en.tsv,fr.tsv}")
    print("  - resources/MANIFEST.json")

if __name__ == "__main__":
    main()
