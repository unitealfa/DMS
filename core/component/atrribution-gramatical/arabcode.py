import os
import re
import sys
from typing import List, Tuple, Dict, Optional


LOCAL_DEPS_DIR = os.path.join(os.getcwd(), ".pylibs")
if os.path.isdir(LOCAL_DEPS_DIR) and LOCAL_DEPS_DIR not in sys.path:
    sys.path.insert(0, LOCAL_DEPS_DIR)

def _install_help():
    py = sys.executable
    print("\n[install help] Sans venv, local dans .pylibs (utilise CE python):")
    print(f'  "{py}" -m pip install --upgrade --no-user-cfg --target "{LOCAL_DEPS_DIR}" camel-tools transformers tokenizers')
    print(f'  "{py}" -m camel_tools.cli.camel_data -i morphology-db-msa-r13')
    print(f'  "{py}" -m camel_tools.cli.camel_data -i ner-arabert')
    print("\nSi tu as lâ€™erreur: Can not combine '--user' and '--target'")
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

# === PONCTUATION ET SYMBOLES: TOUJOURS tag PUNCT + lemma âˆ… ===
# Standard: . , ; : ! ? ( ) [ ] { } < > " '
# Guillemets: " ' " ' Â« Â»
# Tirets: â€” â€“ -
# Arabe: ØŒ Ø› ØŸ Ùª
# Symboles: @ / \ | â€¦ Ù€ % & = + * ^ ~ `
PUNCT_SET = set(list(".,;:!?()[]{}<>\"'""''Â«Â»â€¦â€”â€“-")) | {"ØŒ", "Ø›", "ØŸ", "Ùª", "%", "Ù€", "/", "\\", "|", "@", "â€¦", "&", "=", "+", "*", "^", "~", "`"}
TOKEN_RE = re.compile(rf"({AR_WORD}|[0-9]+|[A-Za-z]+(?:['â€™\-][A-Za-z]+)*|[^\s])", re.UNICODE)

# Principaux codes monétaires ISO 4217 (top 15 demandés, incl. DZD)
CURRENCY_CODES = {
    "USD","EUR","GBP","JPY","CNY","CHF","CAD","AUD","NZD","SEK",
    "NOK","DKK","SAR","AED","DZD"
}
# Symboles usuels mappés sur ces codes
CURRENCY_SYMBOLS = {"\u20ac":"EUR", "$":"USD", "\u00a3":"GBP", "\u00a5":"JPY", "دج":"DZD", "د.ج":"DZD"}
# Noms usuels (normalisés sans diacritiques) pour repérage en lettres
AR_CURRENCY_WORDS = {
    "دينار","دينار جزائري","دنانير",
    "دولار","دولارات",
    "يورو","ين","يوان",
    "درهم","دراهم",
    "ريال","ريالات",
    "فرنك","فرنكات",
    "جنيه","جنيه استرليني","جنيهات"
}

def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text or "")

def is_punct(tok: str) -> bool:
    return tok in PUNCT_SET

# ----------------------------
# 4) Normalization helpers (0-ML)
# ----------------------------
_DIACRITICS_RE = re.compile(rf"[{AR_DIAC}]")
HAMZA_CHARS = set("Ø¡Ø£Ø¥Ø¤Ø¦Ù±")

def strip_diacritics(s: str) -> str:
    return _DIACRITICS_RE.sub("", s or "")

def norm_alef(s: str) -> str:
    return (s or "").replace("Ø£", "Ø§").replace("Ø¥", "Ø§").replace("Ø¢", "Ø§").replace("Ù±", "Ø§")

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
FUT_PARTS = {"Ø³ÙˆÙ"}
NEG_PARTS = {"Ù„Ù…", "Ù„Ù†", "Ù„Ø§", "Ù…Ø§"}
ASPECT_PARTS = {"Ù‚Ø¯"}
Q_PARTS = {"Ù‡Ù„"}
REL_PRON = {"Ø§Ù„Ø°ÙŠ", "Ø§Ù„ØªÙŠ", "Ø§Ù„Ø°ÙŠÙ†", "Ø§Ù„Ù„Ø°ÙŠÙ†", "Ø§Ù„Ù„Ø°Ø§Ù†", "Ø§Ù„Ù„ØªØ§Ù†", "Ø§Ù„Ù„Ø§ØªÙŠ", "Ø§Ù„Ù„ÙˆØ§ØªÙŠ"}
DEM_WORDS = {"Ù‡Ø°Ø§", "Ù‡Ø°Ù‡", "Ù‡Ø¤Ù„Ø§Ø¡", "Ø°Ù„Ùƒ", "ØªÙ„Ùƒ", "Ù‡Ù°Ø°Ø§", "Ù‡Ù°Ø°Ù‡"}
PREP_WORDS_EXT = {"Ø¥Ù„Ù‰", "ÙÙŠ", "Ø¹Ù„Ù‰", "Ù…Ù†", "Ø¹Ù†", "Ù…Ø¹", "Ø­ØªÙ‰", "Ø¹Ø¨Ø±", "Ø¨ÙŠÙ†", "Ù‚Ø¨Ù„", "Ø¨Ø¹Ø¯", "Ø¯ÙˆÙ†", "Ø­ÙˆÙ„", "Ø¹Ù†Ø¯", "Ù„Ø¯Ù‰", "Ù…Ø«Ù„", "Ø®Ù„Ø§Ù„"}
CONJ_WORDS = {"Ùˆ", "Ù", "Ø«Ù…", "Ø£Ùˆ", "Ù„ÙƒÙ†", "Ø¨Ù„", "Ø£Ù…"}
NEG_ADJ = {"ØºÙŠØ±"}
FIX_NOUNS = {"Ø¨Ø¹Ø¶"}

RX_INNA_CLITIC = re.compile(r"^(Ø¥Ù†|Ø£Ù†|Ù„Ø£Ù†)(Ù‡|Ù‡Ø§|Ù‡Ù…|Ù‡Ù†|ÙƒÙ…Ø§|ÙƒÙ…|ÙƒÙ†|Ù†Ø§)?$", re.UNICODE)
RX_LAKIN_CLITIC = re.compile(r"^Ù„ÙƒÙ†(Ù‡|Ù‡Ø§|Ù‡Ù…|Ù‡Ù†|ÙƒÙ…Ø§|ÙƒÙ…|ÙƒÙ†|Ù†Ø§)?$", re.UNICODE)

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
        return ("CC", "Ù„ÙƒÙ†")

    return None

# ----------------------------
# 7) Segmentation clitiques (fix global)
# ----------------------------
PROCLITIC_CONJ = {"Ùˆ", "Ù"}
PROCLITIC_PREP = {"Ø¨", "Ùƒ", "Ù„"}
FUT_CLITIC = {"Ø³"}  # Ø³Ù€ (future)

# suffix pronouns (triÃ©s par longueur)
PRON_SUFFIXES = [
    "ÙƒÙ…Ø§", "ÙƒÙ…", "ÙƒÙ†",
    "Ù‡Ù…Ø§", "Ù‡Ù…", "Ù‡Ù†",
    "Ù‡Ø§", "Ù‡",
    "Ù†Ø§", "Ù†ÙŠ",
    "ÙŠ", "Ùƒ"
]

def _looks_like_verb_imperfect(core: str) -> bool:
    # ÙŠ/Øª/Ø£/Ù† + au moins 3 lettres ensuite
    return bool(core) and core[0] in {"ÙŠ", "Øª", "Ø£", "Ù†"} and len(core) >= 4

def _safe_split_conj(tok0: str) -> bool:
    # split Ùˆ/Ù seulement si reste commence par:
    # - prÃ©position fermÃ©e (Ù…Ù†/ÙÙŠ/Ø¹Ù„Ù‰/Ø¹Ù†/Ø¥Ù„Ù‰/...)
    # - Ø§Ù„...
    # - verbe imperfect (ÙŠ/Øª/Ø£/Ù†...)
    if len(tok0) < 3:
        return False
    rem = tok0[1:]
    if rem.startswith("Ø§Ù„"):
        return True
    if rem in PREP_WORDS_EXT:
        return True
    if any(rem.startswith(x) for x in ("Ù…Ù†", "ÙÙŠ", "Ø¹Ù„Ù‰", "Ø¹Ù†", "Ø¥Ù„Ù‰")):
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
    sa = set(a) - set("Ù€")
    sb = set(b) - set("Ù€")
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
    definite_article = w.startswith("Ø§Ù„")
    imperfect_prefix = _looks_like_verb_imperfect(w)
    looks_like_perfect = (w_len in {3, 4}) and (not imperfect_prefix) and (not definite_article)

    # patterns helpful for verbs like ÙŠÙŽØ±ÙˆØ§ / ÙŠØ±ÙˆÙ†
    ends_verb_plural = w.endswith("ÙˆØ§") or w.endswith("ÙˆÙ†") or w.endswith("ÙŠÙ†")

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
        if nw.startswith("Ø§Ù„") and pos == "verb":
            score += 1

        # prefer lemma similar to surface
        score += int(10 * _char_overlap_ratio(lem_norm, w_norm))

        # penalty: Ø´Ø¯Ø© ÙÙŠ lemma Ø¨Ø¯ÙˆÙ† Ø´Ø¯Ø© ÙÙŠ surface (fix Ù…ØµØ± vs Ù…ÙØµÙØ±Ù‘ et hallucinations)
        if "Ù‘" in (lem or "") and "Ù‘" not in (w_raw or ""):
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
                if "Ù‘" in (lem or "") and "Ù‘" not in (w_raw or ""):
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

    # 1) Ùˆ/Ù (conj) only if safe
    if t0 and t0[0] in PROCLITIC_CONJ and _safe_split_conj(t0):
        prefixes.append((t0[0], "CC", t0[0]))
        t0 = t0[1:]

    # 2) Ø³ (future) if followed by imperfect verb
    if t0 and t0[0] in FUT_CLITIC and len(t0) >= 3 and _looks_like_verb_imperfect(t0[1:]):
        prefixes.append((t0[0], "RP", t0[0]))
        t0 = t0[1:]

    # 3) Ø¨/Ùƒ/Ù„ (prep) : split if remainder has analyses as noun/prop/adj OR starts with Ø§Ù„ OR is known prep base
    if t0 and t0[0] in PROCLITIC_PREP and len(t0) >= 3:
        rem = t0[1:]
        do_split = False
        if rem.startswith("Ø§Ù„") or rem in PREP_WORDS_EXT:
            do_split = True
        else:
            # try analyze remainder; if it yields noun/noun_prop/adj => split
            ans = _best_analysis_for(rem)
            if any((a.get("pos") or "").lower() in {"noun", "noun_prop", "adj"} for a in ans):
                # BUT avoid splitting if whole word is a strong noun_prop match (e.g., Ø¨Ø´Ø§Ø±)
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

        # special connector "Ùˆ" before a pronoun like ...ÙˆÙ‡Ø§ / ...ÙˆÙ‡Ù… / ...ÙˆÙƒÙ…
        if stem.endswith("ÙˆÙ‡Ø§"):
            stem = stem[:-3]
            suffixes.insert(0, ("Ùˆ", "PRP", "Ùˆ"))
            suffixes.insert(1, ("Ù‡Ø§", "PRP", "Ù‡Ø§"))
            changed = True
            continue
        if stem.endswith("ÙˆÙ‡Ù…"):
            stem = stem[:-3]
            suffixes.insert(0, ("Ùˆ", "PRP", "Ùˆ"))
            suffixes.insert(1, ("Ù‡Ù…", "PRP", "Ù‡Ù…"))
            changed = True
            continue
        if stem.endswith("ÙˆÙƒÙ…"):
            stem = stem[:-3]
            suffixes.insert(0, ("Ùˆ", "PRP", "Ùˆ"))
            suffixes.insert(1, ("ÙƒÙ…", "PRP", "ÙƒÙ…"))
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
    "Ø³ÙŠØ¯", "Ø§Ù„Ø³ÙŠØ¯",
    "Ø³ÙŠØ¯Ø©", "Ø§Ù„Ø³ÙŠØ¯Ø©",
    "Ø¯ÙƒØªÙˆØ±", "Ø§Ù„Ø¯ÙƒØªÙˆØ±",
    "Ø¯ÙƒØªÙˆØ±Ø©", "Ø§Ù„Ø¯ÙƒØªÙˆØ±Ø©",
    "Ø§Ø³ØªØ§Ø°", "Ø§Ù„Ø£Ø³ØªØ§Ø°", "Ø§Ù„Ø§Ø³ØªØ§Ø°", "Ø§Ø³ØªØ§Ø°Ø©", "Ø§Ù„Ø£Ø³ØªØ§Ø°Ø©", "Ø§Ù„Ø§Ø³ØªØ§Ø°Ø©",
    "Ù…Ù‡Ù†Ø¯Ø³", "Ø§Ù„Ù…Ù‡Ù†Ø¯Ø³",
    "Ø´ÙŠØ®", "Ø§Ù„Ø´ÙŠØ®",
    "Ø±Ø¦ÙŠØ³", "Ø§Ù„Ø±Ø¦ÙŠØ³",
    "ÙˆØ²ÙŠØ±", "Ø§Ù„ÙˆØ²ÙŠØ±",
    "Ø§Ù…ÙŠØ±", "Ø§Ù„Ø£Ù…ÙŠØ±", "Ø§Ù„Ø§Ù…ÙŠØ±",
    "Ù…Ù„Ùƒ", "Ø§Ù„Ù…Ù„Ùƒ",
}

def norm_for_match(tok: str) -> str:
    return norm_alef(strip_diacritics(tok))

def fix_titles(tokens: List[str], labels: List[str]) -> List[str]:
    labels = labels[:]
    title_set = set(TITLE_NORMS) | set(HONORIFICS_AR)

    for i in range(len(tokens)):
        cur_norm = norm_for_match(tokens[i]).lower()
        cur_norm2 = cur_norm[2:] if cur_norm.startswith("Ø§Ù„") and len(cur_norm) > 2 else cur_norm

        # Cas 1: titre juste avant un B-PERS => fusionner
        if labels[i].startswith("B-PERS") and i > 0:
            prev_norm = norm_for_match(tokens[i - 1]).lower()
            prev_norm2 = prev_norm[2:] if prev_norm.startswith("Ø§Ù„") and len(prev_norm) > 2 else prev_norm
            if prev_norm in title_set or prev_norm2 in title_set:
                labels[i - 1] = "B-PERS"
                labels[i] = "I-PERS"

        # Cas 2: token est un titre, on l'étend avec le nom qui suit (même si non taggué)
        if cur_norm in title_set or cur_norm2 in title_set:
            labels[i] = "B-PERS"
            if i + 1 < len(tokens) and not is_punct(tokens[i + 1]):
                if labels[i + 1].startswith("B-PERS") or labels[i + 1].startswith("I-PERS"):
                    labels[i + 1] = "I-PERS"
                else:
                    labels[i + 1] = "I-PERS"

    return enforce_bio(labels)

def add_year_dates(tokens: List[str], labels: List[str]) -> List[str]:
    labels = labels[:]
    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) == 4:
            y = int(tok)
            if 1500 <= y <= 2100 and labels[i] == "O":
                labels[i] = "B-DATE"
    return labels

def add_currency_codes(tokens: List[str], labels: List[str]) -> List[str]:
    labels = labels[:]
    for i, tok in enumerate(tokens):
        raw = (tok or "").strip()
        if not raw:
            continue
        up = raw.upper()
        if up in CURRENCY_CODES and labels[i] == "O":
            labels[i] = "B-CUR"
            continue
        if raw in CURRENCY_SYMBOLS and labels[i] == "O":
            labels[i] = "B-CUR"
            continue
        norm = norm_alef(strip_diacritics(raw)).lower()
        if norm in AR_CURRENCY_WORDS and labels[i] == "O":
            labels[i] = "B-CUR"
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

# Anti-faux PERS : titres/abréviations courants (liste locale)
HONORIFICS_AR = {
    "آنسـة","آنسة","أ","أ.","أ/","أستاذ","أستاذة","أساتذة","أسقف","أطبـاء","أطباء","أمير","أميرة","إمام","ابونا",
    "اساتذة","اطباء","الآنسة","الأب","الأخ","الأخت","الأستاذ","الأستاذة","الأسقف","الأم","الأم الرئيسة","الأمير",
    "الأميرة","الأمين العام","الإمام","الاستاذ","الاستاذة","الامير","البابا","الجنرال","الحاج","الحاجة","الحاخام",
    "الدكتور","الدكتورة","الرئيس","الرئيس التنفيذي","الرئيس التنفيذي للعمليات","الرئيسة","الرائد","الراعي","الرقيب",
    "السفير","السفيرة","السيد","السيدة","الشيخ","الشيخة","الصيدلانية","الصيدلي","الضابط","العقيد","العميد","القائد",
    "القاضي","القاضية","القس","القسيس","القنصل","الكابتن","المحافظ","المحامي","المحامية","المحترم","المحترمة",
    "المدير","المدير التقني","المدير التنفيذي","المدير المالي","المديرة","المستشار","المفتش","المفتي","الملازم",
    "الملك","الملكة","المهندس","المهندسة","الوالي","الوزير","الوزير الأول","الوزيرة","امام","انسة","بروف","بروف.",
    "بروفيسور","بيطري","جلالة","جنرال","حاج","حاجة","حاخام","حضرة","د","د.","د/","دكاترة","دكاتره","دكتور",
    "دكتور أسنان","دكتور بيطري","دكتورة","دكتورة أسنان","رئيس","رئيس الأساقفة","رئيس البلدية","رئيس الجامعة",
    "رئيس الحكومة","رئيس الوزراء","رئيس قسم","رئيس مجلس الإدارة","رئيس مصلحة","رائد","راعي","راهبة","رقيب","سعادة",
    "سمو","سمو الأمير","سمو الأميرة","سيد","سيدة","سيدي","سير","سيناتور","سيّد","سيّدة","شيخ","صاحب الجلالة",
    "صاحب السمو","صاحب الفخامة","صاحبة الجلالة","صاحبة السمو","صيدلانية","صيدلي","ضابط","طبيب أسنان","طبيب بيطري",
    "طبيبة أسنان","عضو البرلمان","عضو مجلس الشيوخ","عضو مجلس النواب","عقيد","عمدة","عميد","عون","فخامة",
    "فخامة الرئيس","قائد","قاض","قاضي","قنصل","كابتن","كاتب الدولة","كاردينال","لالة","لواء","م","م.","م/","محافظ",
    "محام","محامية","مدير","مدير التكنولوجيا","مدير العمليات","مديرة","مسؤول","مستشار","مشرف","مشير","معالي",
    "معماري","مفتش","مفتي","مفوض","مفوّض","ملازم","ملك","ملكة","مهندس","مهندس معماري","مهندسة","موثق","موثقة",
    "مولاي","نائب","نائب الرئيس","نائبة","نائبة الرئيس","نقيب","والي","وزير دولة","وكيل",
    "امير","اميرة","الامير","الاميرة","ام","الأم","الأخت","الأمير","الأميرة","السادة","السيدات","ام","الأم","الأخت"
}

def _is_initial_chain(tokens: List[str]) -> bool:
    letters = [t for t in tokens if re.fullmatch(r"[A-Za-z]", t)]
    dots = [t for t in tokens if t == "."]
    long = any(len(t) > 2 and t.isalpha() for t in tokens)
    return (len(letters) >= 2) and (len(dots) >= len(letters) - 1) and not long

def drop_false_pers(tokens: List[str], labels: List[str]) -> List[str]:
    out = labels[:]
    i = 0
    while i < len(out):
        lab = out[i]
        if not lab.startswith("B-"):
            i += 1
            continue
        typ = lab[2:]
        j = i + 1
        while j < len(out) and out[j] == f"I-{typ}":
            j += 1
        span_tokens = tokens[i:j]
        if typ == "PERS":
            if _is_initial_chain(span_tokens):
                for k in range(i, j):
                    out[k] = "O"
            elif all(strip_diacritics(t).lower() in HONORIFICS_AR for t in span_tokens):
                for k in range(i, j):
                    out[k] = "O"
        i = j
    return enforce_bio(out)

def strip_entity_clitics_for_display(tok: str, lemma: str) -> str:
    t = strip_diacritics(tok)
    l = strip_diacritics(lemma or "")

    if len(t) > 2 and (t.startswith("ÙˆØ§Ù„") or t.startswith("ÙØ§Ù„")):
        return tok[1:]

    if len(t) > 1 and t.startswith("Ø¨"):
        rem = t[1:]
        rem2 = rem[2:] if rem.startswith("Ø§Ù„") and len(rem) > 2 else rem
        l2 = l[2:] if l.startswith("Ø§Ù„") and len(l) > 2 else l
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
    if undiac.startswith(("Ùˆ", "Ù")) and len(undiac) > 2 and _safe_split_conj(undiac):
        c += 1
        undiac = undiac[1:]
    if undiac.startswith(("Ø¨", "Ùƒ", "Ù„")) and len(undiac) > 2:
        c += 1
        undiac = undiac[1:]
    # suffix pronouns
    for suf in PRON_SUFFIXES:
        if undiac.endswith(suf):
            c += 1
            break
    if undiac.endswith(("ÙˆÙ‡Ø§", "ÙˆÙ‡Ù…", "ÙˆÙƒÙ…")):
        c += 2
    return c

def _lemma_suspicious(word_raw: str, lemma: str) -> bool:
    w = norm_alef(strip_diacritics(word_raw))
    l = norm_alef(strip_diacritics(lemma or ""))
    if not w or not l:
        return False
    # suspicious if lemma contains shadda but word doesn't
    if "Ù‘" in (lemma or "") and "Ù‘" not in (word_raw or ""):
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
            display_rows.append((tok, "PUNCT", "âˆ…"))
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

        # Analyze STEM only (important fix for Ø¨Ù…ØµØ±, ÙˆÙ…Ù†Ù‡Ù…, ... pronoun verbs)
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
        ner_labels = add_currency_codes(toks, ner_labels)
        ner_labels = enforce_bio(ner_labels)
        ner_labels = drop_false_pers(toks, ner_labels)

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
    parts = re.split(r"[\.\!\ØŸ\?]+", s)
    return [p.strip() for p in parts if p.strip()]

# (ancienne invite supprimÃ©e pour Ã©viter le bruit au chargement)

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

