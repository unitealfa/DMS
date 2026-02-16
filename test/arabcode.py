import os
import re
import sys
from typing import List, Tuple, Dict, Optional


LOCAL_DEPS_DIR = os.path.join(os.getcwd(), ".pylibs")
if os.path.isdir(LOCAL_DEPS_DIR) and LOCAL_DEPS_DIR not in sys.path:
    sys.path.insert(0, LOCAL_DEPS_DIR)

print("Python:", sys.executable)
print("Local deps:", LOCAL_DEPS_DIR if os.path.isdir(LOCAL_DEPS_DIR) else "(absent)")

def _install_help():
    py = sys.executable
    print("\n[install help] Sans venv, local dans .pylibs (utilise CE python):")
    print(f'  "{py}" -m pip install --upgrade --no-user-cfg --target "{LOCAL_DEPS_DIR}" camel-tools transformers tokenizers')
    print(f'  "{py}" -m camel_tools.cli.camel_data -i morphology-db-msa-r13')
    print(f'  "{py}" -m camel_tools.cli.camel_data -i ner-arabert')
    print("\nSi tu as l’erreur: Can not combine '--user' and '--target'")
    print("  PowerShell:")
    print('    $env:PIP_CONFIG_FILE="NUL"')
    print("    Remove-Item Env:PIP_USER -ErrorAction SilentlyContinue")
    print('    py -3.11 -m pip install --upgrade --no-user-cfg --target .\\.pylibs camel-tools')

# ----------------------------
# 1) Imports CAMeL
# ----------------------------
try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
except Exception as e:
    print("\n[error] camel_tools not available:", e)
    _install_help()
    raise

# NER pretrained
try:
    from camel_tools.ner import NERecognizer
except Exception:
    NERecognizer = None

# ----------------------------
# 2) Build analyzer (requires morphology-db-msa-r13)
# ----------------------------
try:
    DB = MorphologyDB.builtin_db()
except FileNotFoundError as e:
    raise FileNotFoundError(
        "CAMeL morphology DB not found. Run:\n"
        f'  "{sys.executable}" -m camel_tools.cli.camel_data -i morphology-db-msa-r13\n'
    ) from e

ANALYZER = Analyzer(DB, backoff="NOAN_PROP")

# ----------------------------
# 3) Tokenization (Arabic words + digits + Latin + single-char punctuation)
# ----------------------------
AR_BASE = r"\u0621-\u063A\u0641-\u064A\u066E-\u066F\u0671-\u06D3\u06FA-\u06FC"
AR_DIAC = r"\u064B-\u065F\u0670\u0640"  # harakat + shadda + superscript alef + tatweel
AR_WORD = rf"(?:[{AR_BASE}][{AR_DIAC}]*)+"

# === PONCTUATION ET SYMBOLES: TOUJOURS tag PUNCT + lemma ∅ ===
# Standard: . , ; : ! ? ( ) [ ] { } < > " '
# Guillemets: " ' " ' « »
# Tirets: — – -
# Arabe: ، ؛ ؟ ٪
# Symboles: @ / \ | … ـ % & = + * ^ ~ `
PUNCT_SET = set(list(".,;:!?()[]{}<>\"'""''«»…—–-")) | {"،", "؛", "؟", "٪", "%", "ـ", "/", "\\", "|", "@", "…", "&", "=", "+", "*", "^", "~", "`"}
TOKEN_RE = re.compile(rf"({AR_WORD}|[0-9]+|[A-Za-z]+(?:['’\-][A-Za-z]+)*|[^\s])", re.UNICODE)

def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text or "")

def is_punct(tok: str) -> bool:
    return tok in PUNCT_SET

# ----------------------------
# 4) Normalization helpers (0-ML)
# ----------------------------
_DIACRITICS_RE = re.compile(rf"[{AR_DIAC}]")
HAMZA_CHARS = set("ءأإؤئٱ")

def strip_diacritics(s: str) -> str:
    return _DIACRITICS_RE.sub("", s or "")

def norm_alef(s: str) -> str:
    return (s or "").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")

def has_hamza(s: str) -> bool:
    return any(ch in HAMZA_CHARS for ch in (s or ""))

# ----------------------------
# 5) CAMeL POS -> Penn-like
# ----------------------------
AR_POS2PENN = {
    "noun": "NN",
    "noun_prop": "NNP",
    "verb": "VB",
    "adj": "JJ",
    "adv": "RB",
    "prep": "IN",
    "conj": "CC",
    "pron": "PRP",
    "pron_dem": "PRP",
    "det": "DT",
    "num": "CD",
    "part": "RP",
    "part_fut": "RP",
    "abbrev": "NN",
}

def penn_from_analysis(a: Dict) -> str:
    pos = (a.get("pos") or "").lower()
    return AR_POS2PENN.get(pos, "NN")

def lemma_from_analysis(a: Dict, fallback_word: str) -> str:
    return a.get("lex") or a.get("lemma") or fallback_word

# ----------------------------
# 6) Strong closed-class overrides (0-ML)
# ----------------------------
FUT_PARTS = {"سوف"}
NEG_PARTS = {"لم", "لن", "لا", "ما"}
ASPECT_PARTS = {"قد"}
Q_PARTS = {"هل"}
REL_PRON = {"الذي", "التي", "الذين", "اللذين", "اللذان", "اللتان", "اللاتي", "اللواتي"}
DEM_WORDS = {"هذا", "هذه", "هؤلاء", "ذلك", "تلك", "هٰذا", "هٰذه"}
PREP_WORDS_EXT = {"إلى", "في", "على", "من", "عن", "مع", "حتى", "عبر", "بين", "قبل", "بعد", "دون", "حول", "عند", "لدى", "مثل", "خلال"}
CONJ_WORDS = {"و", "ف", "ثم", "أو", "لكن", "بل", "أم"}
NEG_ADJ = {"غير"}
FIX_NOUNS = {"بعض"}

RX_INNA_CLITIC = re.compile(r"^(إن|أن|لأن)(ه|ها|هم|هن|كما|كم|كن|نا)?$", re.UNICODE)
RX_LAKIN_CLITIC = re.compile(r"^لكن(ه|ها|هم|هن|كما|كم|كن|نا)?$", re.UNICODE)

def override_tag(tok: str) -> Optional[Tuple[str, str]]:
    if not tok or is_punct(tok):
        return None

    t0 = strip_diacritics(tok)

    if t0 in FIX_NOUNS:
        return ("NN", t0)

    if t0 in CONJ_WORDS:
        return ("CC", t0)

    if t0 in FUT_PARTS or t0 in NEG_PARTS or t0 in ASPECT_PARTS or t0 in Q_PARTS:
        return ("RP", t0)

    if t0 in REL_PRON:
        return ("WDT", t0)

    if t0 in DEM_WORDS:
        return ("PRP", t0)

    if t0 in PREP_WORDS_EXT:
        return ("IN", t0)

    if t0 in NEG_ADJ:
        return ("JJ", t0)

    m = RX_INNA_CLITIC.match(t0)
    if m:
        return ("RP", m.group(1))

    m2 = RX_LAKIN_CLITIC.match(t0)
    if m2:
        return ("CC", "لكن")

    return None

# ----------------------------
# 7) Segmentation clitiques (fix global)
# ----------------------------
PROCLITIC_CONJ = {"و", "ف"}
PROCLITIC_PREP = {"ب", "ك", "ل"}
FUT_CLITIC = {"س"}  # سـ (future)

# suffix pronouns (triés par longueur)
PRON_SUFFIXES = [
    "كما", "كم", "كن",
    "هما", "هم", "هن",
    "ها", "ه",
    "نا", "ني",
    "ي", "ك"
]

def _looks_like_verb_imperfect(core: str) -> bool:
    # ي/ت/أ/ن + au moins 3 lettres ensuite
    return bool(core) and core[0] in {"ي", "ت", "أ", "ن"} and len(core) >= 4

def _safe_split_conj(tok0: str) -> bool:
    # split و/ف seulement si reste commence par:
    # - préposition fermée (من/في/على/عن/إلى/...)
    # - ال...
    # - verbe imperfect (ي/ت/أ/ن...)
    if len(tok0) < 3:
        return False
    rem = tok0[1:]
    if rem.startswith("ال"):
        return True
    if rem in PREP_WORDS_EXT:
        return True
    if any(rem.startswith(x) for x in ("من", "في", "على", "عن", "إلى")):
        return True
    if _looks_like_verb_imperfect(rem):
        return True
    return False

def _best_analysis_for(form_undiac: str) -> List[Dict]:
    try:
        return ANALYZER.analyze(form_undiac)
    except Exception:
        return []

def _char_overlap_ratio(a: str, b: str) -> float:
    sa = set(a) - set("ـ")
    sb = set(b) - set("ـ")
    if not sa:
        return 0.0
    return len(sa & sb) / max(1, len(sa))

def pick_analysis(
    word_raw: str,
    analyses: List[Dict],
    prev_tag: Optional[str] = None,
    prev_word: Optional[str] = None,
    next_word: Optional[str] = None,
    is_first_content: bool = False
) -> Optional[Dict]:
    if not analyses:
        return None

    w_raw = word_raw or ""
    w = strip_diacritics(w_raw)
    w_norm = norm_alef(w)

    nw = strip_diacritics(next_word or "")
    pw = strip_diacritics(prev_word or "")

    w_len = len(w)
    definite_article = w.startswith("ال")
    imperfect_prefix = _looks_like_verb_imperfect(w)
    looks_like_perfect = (w_len in {3, 4}) and (not imperfect_prefix) and (not definite_article)

    # patterns helpful for verbs like يَروا / يرون
    ends_verb_plural = w.endswith("وا") or w.endswith("ون") or w.endswith("ين")

    has_simple_verb = any(((a.get("pos") or "").lower() == "verb") for a in analyses)

    best = None
    best_score = -10**9

    for a in analyses:
        pos = (a.get("pos") or "").lower()
        lem = lemma_from_analysis(a, w_raw)
        lem0 = strip_diacritics(lem)
        lem_norm = norm_alef(lem0)

        score = 0

        # closed-class boost
        if pos in {"prep", "conj", "det", "pron", "pron_dem", "part", "part_fut", "num"}:
            score += 4

        # strong imperfect preference
        if imperfect_prefix:
            if pos == "verb":
                score += 10
            else:
                score -= 6

        if ends_verb_plural:
            if pos == "verb":
                score += 6
            else:
                score -= 4

        # context after preposition
        if prev_tag == "IN":
            if pos in {"noun", "adj", "noun_prop"}:
                score += 4
            if pos == "verb":
                score -= 2

        # definite article
        if definite_article:
            if pos in {"noun", "adj"}:
                score += 4
            if pos == "verb":
                score -= 3
            if pos == "noun_prop":
                score -= 2

        # sentence-initial bias
        if is_first_content:
            if pos == "verb":
                score += 2
            if pos == "noun_prop":
                score -= 1

        # if next word has definite article, current verb less likely (weak)
        if nw.startswith("ال") and pos == "verb":
            score += 1

        # prefer lemma similar to surface
        score += int(10 * _char_overlap_ratio(lem_norm, w_norm))

        # penalty: شدة في lemma بدون شدة في surface (fix مصر vs مُصِرّ et hallucinations)
        if "ّ" in (lem or "") and "ّ" not in (w_raw or ""):
            score -= 6

        # verbs: sanity
        if pos == "verb":
            if looks_like_perfect and lem_norm == w_norm:
                score += 6
            # hamza consistency (mild)
            if has_hamza(w_raw) and not has_hamza(lem):
                score -= 2

        # noun_prop: prefer when lemma exactly matches surface (proper nouns)
        if pos == "noun_prop" and lem_norm == w_norm:
            score += 8

        # if we have a verb candidate set, penalize rare non-verb picks for imperfect
        if has_simple_verb and imperfect_prefix and pos in {"noun_prop", "noun"}:
            score -= 3

        if score > best_score:
            best_score = score
            best = a

    # last-ditch: if selected not verb but word clearly imperfect and there exists verb analysis => pick best verb
    if imperfect_prefix and best is not None and (best.get("pos","").lower() != "verb"):
        verb_as = [a for a in analyses if (a.get("pos") or "").lower() == "verb"]
        if verb_as:
            vb = None
            vb_sc = -10**9
            for a in verb_as:
                lem = lemma_from_analysis(a, w_raw)
                lem_norm = norm_alef(strip_diacritics(lem))
                sc = int(10 * _char_overlap_ratio(lem_norm, w_norm))
                if "ّ" in (lem or "") and "ّ" not in (w_raw or ""):
                    sc -= 6
                if sc > vb_sc:
                    vb_sc = sc
                    vb = a
            if vb is not None:
                best = vb

    return best

def split_proclitics(tok: str) -> Tuple[List[Tuple[str,str,str]], str]:
    """
    Return (prefix_morphemes, core_undiac)
    prefix morphemes: (surface, tag, lemma)
    """
    t0 = strip_diacritics(tok)
    prefixes: List[Tuple[str,str,str]] = []

    # 1) و/ف (conj) only if safe
    if t0 and t0[0] in PROCLITIC_CONJ and _safe_split_conj(t0):
        prefixes.append((t0[0], "CC", t0[0]))
        t0 = t0[1:]

    # 2) س (future) if followed by imperfect verb
    if t0 and t0[0] in FUT_CLITIC and len(t0) >= 3 and _looks_like_verb_imperfect(t0[1:]):
        prefixes.append((t0[0], "RP", t0[0]))
        t0 = t0[1:]

    # 3) ب/ك/ل (prep) : split if remainder has analyses as noun/prop/adj OR starts with ال OR is known prep base
    if t0 and t0[0] in PROCLITIC_PREP and len(t0) >= 3:
        rem = t0[1:]
        do_split = False
        if rem.startswith("ال") or rem in PREP_WORDS_EXT:
            do_split = True
        else:
            # try analyze remainder; if it yields noun/noun_prop/adj => split
            ans = _best_analysis_for(rem)
            if any((a.get("pos") or "").lower() in {"noun", "noun_prop", "adj"} for a in ans):
                # BUT avoid splitting if whole word is a strong noun_prop match (e.g., بشار)
                whole = _best_analysis_for(t0)
                best_whole = pick_analysis(tok, whole)
                if not (best_whole and (best_whole.get("pos","").lower()=="noun_prop") and
                        (norm_alef(strip_diacritics(lemma_from_analysis(best_whole, t0))) == norm_alef(rem) or
                         norm_alef(strip_diacritics(lemma_from_analysis(best_whole, t0))) == norm_alef(t0))):
                    do_split = True

        if do_split:
            prefixes.append((t0[0], "IN", t0[0]))  # display as preposition
            t0 = rem

    return prefixes, t0

def split_enclitics(core_undiac: str) -> Tuple[str, List[Tuple[str,str,str]]]:
    """
    Split suffix pronouns from the right.
    Return (stem, suffix_morphemes) where suffix are in correct reading order.
    """
    stem = core_undiac
    suffixes: List[Tuple[str,str,str]] = []

    # iterative stripping: longest first
    changed = True
    while changed and stem:
        changed = False

        # special connector "و" before a pronoun like ...وها / ...وهم / ...وكم
        if stem.endswith("وها"):
            stem = stem[:-3]
            suffixes.insert(0, ("و", "PRP", "و"))
            suffixes.insert(1, ("ها", "PRP", "ها"))
            changed = True
            continue
        if stem.endswith("وهم"):
            stem = stem[:-3]
            suffixes.insert(0, ("و", "PRP", "و"))
            suffixes.insert(1, ("هم", "PRP", "هم"))
            changed = True
            continue
        if stem.endswith("وكم"):
            stem = stem[:-3]
            suffixes.insert(0, ("و", "PRP", "و"))
            suffixes.insert(1, ("كم", "PRP", "كم"))
            changed = True
            continue

        for suf in PRON_SUFFIXES:
            if stem.endswith(suf) and len(stem) > len(suf) + 1:
                stem = stem[:-len(suf)]
                suffixes.insert(0, (suf, "PRP", suf))
                changed = True
                break

    return stem, suffixes

# -------------------- NER helpers (pretrained + deterministic fixes) --------------------
_NER = None

def get_ner():
    global _NER
    if _NER is not None:
        return _NER
    if NERecognizer is None:
        raise ImportError(
            "camel_tools.ner is not available.\n"
            "Install and download data:\n"
            f'  "{sys.executable}" -m pip install --upgrade --no-user-cfg --target "{LOCAL_DEPS_DIR}" camel-tools transformers tokenizers\n'
            f'  "{sys.executable}" -m camel_tools.cli.camel_data -i ner-arabert\n'
        )
    _NER = NERecognizer.pretrained()
    return _NER

def ner_predict_tokens(tokens: List[str]) -> List[str]:
    ner = get_ner()
    if hasattr(ner, "predict_sentence"):
        return ner.predict_sentence(tokens)
    if hasattr(ner, "predict"):
        return ner.predict(tokens)
    raise AttributeError("NERecognizer has no predict_sentence/predict method in this version.")

TITLE_NORMS = {
    "سيد", "السيد",
    "سيدة", "السيدة",
    "دكتور", "الدكتور",
    "دكتورة", "الدكتورة",
    "استاذ", "الأستاذ", "الاستاذ", "استاذة", "الأستاذة", "الاستاذة",
    "مهندس", "المهندس",
    "شيخ", "الشيخ",
    "رئيس", "الرئيس",
    "وزير", "الوزير",
    "امير", "الأمير", "الامير",
    "ملك", "الملك",
}

def norm_for_match(tok: str) -> str:
    return norm_alef(strip_diacritics(tok))

def fix_titles(tokens: List[str], labels: List[str]) -> List[str]:
    labels = labels[:]
    for i in range(1, len(tokens)):
        if labels[i].startswith("B-PERS") and labels[i - 1] == "O":
            prev_norm = norm_for_match(tokens[i - 1])
            prev_norm2 = prev_norm[2:] if prev_norm.startswith("ال") and len(prev_norm) > 2 else prev_norm
            if prev_norm in TITLE_NORMS or prev_norm2 in TITLE_NORMS:
                labels[i - 1] = "B-PERS"
                labels[i] = "I-PERS"
    return labels

def add_year_dates(tokens: List[str], labels: List[str]) -> List[str]:
    labels = labels[:]
    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) == 4:
            y = int(tok)
            if 1500 <= y <= 2100 and labels[i] == "O":
                labels[i] = "B-DATE"
    return labels

def enforce_bio(labels: List[str]) -> List[str]:
    out = labels[:]
    for i in range(len(out)):
        lab = out[i]
        if lab.startswith("I-"):
            typ = lab[2:]
            if i == 0 or out[i - 1] == "O" or out[i - 1][2:] != typ:
                out[i] = "B-" + typ
    return out

def strip_entity_clitics_for_display(tok: str, lemma: str) -> str:
    t = strip_diacritics(tok)
    l = strip_diacritics(lemma or "")

    if len(t) > 2 and (t.startswith("وال") or t.startswith("فال")):
        return tok[1:]

    if len(t) > 1 and t.startswith("ب"):
        rem = t[1:]
        rem2 = rem[2:] if rem.startswith("ال") and len(rem) > 2 else rem
        l2 = l[2:] if l.startswith("ال") and len(l) > 2 else l
        if l2 == rem2 and rem:
            return tok[1:]

    return tok

def bio_to_entities(tokens: List[str], labels: List[str], morph_rows: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    ents = []
    i = 0
    n = len(tokens)

    def pos_of(k: int) -> str:
        return (morph_rows[k].get("tag") or "").strip()

    def lemma_of(k: int) -> str:
        return (morph_rows[k].get("lemma") or "").strip()

    while i < n:
        lab = labels[i]
        if lab == "O" or "-" not in lab:
            i += 1
            continue
        pref, typ = lab.split("-", 1)
        if pref != "B":
            i += 1
            continue

        j = i + 1
        while j < n and labels[j] == f"I-{typ}":
            j += 1

        start, end = i, j
        while start < end and pos_of(start) in {"IN", "CC", "RP"}:
            start += 1
        while end > start and pos_of(end - 1) in {"IN", "CC", "RP"}:
            end -= 1

        if start < end:
            parts = []
            for k in range(start, end):
                if is_punct(tokens[k]):
                    continue
                parts.append(strip_entity_clitics_for_display(tokens[k], lemma_of(k)))
            text = " ".join(p for p in parts if p)
            if text:
                ents.append((typ, text))

        i = j

    return ents

# -------------------- Audits (heuristiques) --------------------
def _estimate_clitic_count(undiac: str) -> int:
    if not undiac:
        return 0
    c = 0
    if undiac.startswith(("و", "ف")) and len(undiac) > 2 and _safe_split_conj(undiac):
        c += 1
        undiac = undiac[1:]
    if undiac.startswith(("ب", "ك", "ل")) and len(undiac) > 2:
        c += 1
        undiac = undiac[1:]
    # suffix pronouns
    for suf in PRON_SUFFIXES:
        if undiac.endswith(suf):
            c += 1
            break
    if undiac.endswith(("وها", "وهم", "وكم")):
        c += 2
    return c

def _lemma_suspicious(word_raw: str, lemma: str) -> bool:
    w = norm_alef(strip_diacritics(word_raw))
    l = norm_alef(strip_diacritics(lemma or ""))
    if not w or not l:
        return False
    # suspicious if lemma contains shadda but word doesn't
    if "ّ" in (lemma or "") and "ّ" not in (word_raw or ""):
        return True
    # suspicious if overlap too low
    if _char_overlap_ratio(l, w) < 0.34 and len(l) >= 4:
        return True
    return False

# -------------------- Main analysis --------------------
def analyze_sentence(text: str):
    toks = simple_tokenize(text)

    print("\n" + "=" * 90)
    print("INPUT:", text)

    prev_tag = None
    prev_word = ""
    seen_content = False

    # These rows are for NER trimming, so keep 1 row per ORIGINAL token
    morph_rows_for_ner: List[Dict[str, str]] = []

    # For display, we may output multiple rows per token (segmentation)
    display_rows: List[Tuple[str, str, str]] = []

    seg_ok = 0
    seg_err = 0
    lem_susp = 0
    lem_total = 0

    for i, tok in enumerate(toks):
        if is_punct(tok):
            display_rows.append((tok, "PUNCT", "∅"))
            morph_rows_for_ner.append({"tok": tok, "tag": "PUNCT", "lemma": ""})
            prev_tag = None
            prev_word = ""
            continue

        if tok.isdigit():
            display_rows.append((tok, "CD", tok))
            morph_rows_for_ner.append({"tok": tok, "tag": "CD", "lemma": tok})
            prev_tag = "CD"
            prev_word = tok
            seen_content = True
            continue

        # next word (skip punctuation)
        next_word = ""
        for j in range(i + 1, len(toks)):
            if not is_punct(toks[j]):
                next_word = toks[j]
                break

        # Closed-class overrides (single token)
        ov = override_tag(tok)
        if ov is not None:
            tag, lem = ov
            display_rows.append((tok, tag, lem))
            morph_rows_for_ner.append({"tok": tok, "tag": tag, "lemma": lem})
            prev_tag = tag
            prev_word = strip_diacritics(tok)
            seen_content = True
            continue

        # ---- segmentation (fix) ----
        prefixes, core0 = split_proclitics(tok)
        stem0, suffixes = split_enclitics(core0)

        # segmentation audit
        clitic_est = _estimate_clitic_count(strip_diacritics(tok))
        if clitic_est >= 2 and (len(prefixes) + len(suffixes) == 0):
            seg_err += 1
        else:
            seg_ok += 1

        # print prefixes rows
        for (p_surf, p_tag, p_lem) in prefixes:
            display_rows.append((p_surf, p_tag, p_lem))

        # Analyze STEM only (important fix for بمصر, ومنهم, ... pronoun verbs)
        analyze_form = stem0 if stem0 else core0
        analyses = _best_analysis_for(analyze_form)

        best = pick_analysis(
            word_raw=analyze_form,
            analyses=analyses,
            prev_tag=prev_tag,
            prev_word=prev_word,
            next_word=next_word,
            is_first_content=(not seen_content),
        )

        if best is None:
            tag = "NN"
            lemma = analyze_form
        else:
            tag = penn_from_analysis(best)
            lemma = lemma_from_analysis(best, analyze_form)

        display_rows.append((analyze_form if analyze_form else tok, tag, lemma))

        # suffix pronouns (display)
        for (s_surf, s_tag, s_lem) in suffixes:
            display_rows.append((s_surf, s_tag, s_lem))

        # For NER row (1 per original token): store the chosen tag/lemma for the whole token
        # We store the STEM lemma to avoid misleading NER trimming/cleaning.
        morph_rows_for_ner.append({"tok": tok, "tag": tag, "lemma": lemma})

        # lemma audit
        if not is_punct(tok):
            lem_total += 1
            if _lemma_suspicious(tok, lemma):
                lem_susp += 1

        prev_tag = tag
        prev_word = analyze_form
        seen_content = True

    # ---- Display table (possibly morpheme-expanded) ----
    maxw = max(12, max(len(r[0]) for r in display_rows) if display_rows else 12)
    for t, tag, lem in display_rows:
        print(f"{t:>{maxw}}  {tag:<6}  lemma={lem}")

    # ---- NER pretrained + deterministic fixes ----
    try:
        ner_labels = ner_predict_tokens(toks)
        ner_labels = fix_titles(toks, ner_labels)
        ner_labels = add_year_dates(toks, ner_labels)
        ner_labels = enforce_bio(ner_labels)

        print("\nNER (pretrained + deterministic fixes) (token, label):")
        print(list(zip(toks, ner_labels)))

        ents = bio_to_entities(toks, ner_labels, morph_rows_for_ner)
        print("Entities:")
        for typ, txt in ents:
            print(f"  {typ}: {txt}")

    except Exception as e:
        print("\nNER skipped (reason):", str(e))

    # ---- Audits (heuristiques) ----
    total_seg = max(1, seg_ok + seg_err)
    print("\nAudit (heuristique):")
    print(f"  Segmentation: OK={seg_ok} | Err={seg_err} | Err%={(seg_err/total_seg*100):.2f} %")
    if lem_total:
        print(f"  Lemma suspect: {lem_susp}/{lem_total} = {(lem_susp/lem_total*100):.2f} %")
    else:
        print("  Lemma suspect: n/a")

def split_input_into_sentences(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    # Split sur les points, points d'exclamation, points d'interrogation uniquement
    parts = re.split(r"[\.\!\؟\?]+", s)
    return [p.strip() for p in parts if p.strip()]

print("Type an Arabic sentence (empty line to stop). You can paste ONE sentence or MANY quoted sentences.")

def run_one_auto(sentence: str):
    """Run AR pipeline on one sentence."""
    return analyze_sentence(sentence)

def run_from_previous_cell(data=None, max_sentences=None):
    """
    Jupyter-only helper: fetch previous cell output and run AR pipeline
    only on sentences detected as Arabic.
    """
    try:
        from nb_utils import detect_lang, get_previous_cell_input, iter_sentences_from_input
    except Exception as e:
        raise ImportError(
            "nb_utils.py is required for run_from_previous_cell(). "
            "Put nb_utils.py in the same folder as arabcode.py."
        ) from e

    if data is None:
        data = get_previous_cell_input()

    if data is None:
        print("[info] No previous-cell data found (selected / FINAL_DOCS / DOCS / _).")
        return []

    results = []
    count = 0

    for doc_name, page_idx, sent_idx, sent in iter_sentences_from_input(data):
        sent = (sent or "").strip()
        if not sent:
            continue
        if detect_lang(sent) != "ar":
            continue

        print()
        print("#" * 90)
        header = f"DOC={doc_name}"
        if page_idx is not None:
            header += f" | page={page_idx}"
        if sent_idx is not None:
            header += f" | sent={sent_idx}"
        header += " | lang=ar"
        print(header)
        print("#" * 90)

        out = run_one_auto(sent)
        if isinstance(out, dict):
            out = dict(out)
            out["doc"] = doc_name
            out["page"] = page_idx
            out["sent_index"] = sent_idx
        results.append(out)

        count += 1
        if max_sentences is not None and count >= max_sentences:
            break

    return results

def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Arabic POS/Lemma/NER (Notebook-friendly). "
                    "In Jupyter, use --from-previous-cell."
    )
    p.add_argument("--text", type=str, default="", help="Analyze a single sentence.")
    p.add_argument("--from-previous-cell", action="store_true",
                   help="Jupyter: use previous cell output as input (Arabic only).")
    p.add_argument("--max-sentences", type=int, default=None)
    args = p.parse_args(argv)

    if args.from_previous_cell:
        run_from_previous_cell(max_sentences=args.max_sentences)
        return

    if args.text:
        run_one_auto(args.text)
        return

    print("Nothing to do. Provide --text, or run with --from-previous-cell inside Jupyter.")

if __name__ == "__main__":
    main()
