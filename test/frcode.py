import os, re, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ============================================================
# 0) Option deps local (sans venv)
# ============================================================
LOCAL_DEPS_DIR = os.path.join(os.getcwd(), ".pylibs")
if os.path.isdir(LOCAL_DEPS_DIR) and LOCAL_DEPS_DIR not in sys.path:
    sys.path.insert(0, LOCAL_DEPS_DIR)

print("Python kernel:", sys.executable)
print("Local deps dir:", LOCAL_DEPS_DIR if os.path.isdir(LOCAL_DEPS_DIR) else "(absent)")

def print_install_help():
    py = sys.executable
    print("\n[install] Sans venv, en local dans .pylibs (utilise CE python):")
    print(f'  "{py}" -m pip install --upgrade --no-user-cfg --target "{LOCAL_DEPS_DIR}" simplemma')
    print(f'  "{py}" -m pip install --upgrade --no-user-cfg --target "{LOCAL_DEPS_DIR}" torch transformers tokenizers')
    print("\nSi tu as: ERROR: Can not combine '--user' and '--target'")
    print("  PowerShell (désactive config pip qui force --user):")
    print('    $env:PIP_CONFIG_FILE="NUL"')
    print("    Remove-Item Env:PIP_USER -ErrorAction SilentlyContinue")
    print(f'    "{py}" -m pip install --upgrade --no-user-cfg --target "{LOCAL_DEPS_DIR}" torch transformers tokenizers')

# ============================================================
# 1) Lemma (simplemma optionnel, déterministe)
# ============================================================
SIMPLEMMA_OK = False
try:
    import simplemma
    SIMPLEMMA_OK = True
except Exception:
    SIMPLEMMA_OK = False


# ============================================================
# 2) Transformers NER (optionnel) SANS sentencepiece
#    Choix modèle WordPiece: bert-base-multilingual-cased
# ============================================================
HF_OK = False
_HF_IMPORT_ERR = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    HF_OK = True
except Exception as e:
    HF_OK = False
    _HF_IMPORT_ERR = e

HF_MODEL_NAME = "Davlan/bert-base-multilingual-cased-ner-hrl"  # WordPiece (mBERT), pas sentencepiece


# ============================================================
# 3) Tokenisation robuste FR (+ offsets) + élisions
# ============================================================
URL_RE   = r"""(?:https?://|www\.)[^\s<>"']+"""
EMAIL_RE = r"""[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"""
IPV4_RE  = r"""\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?\b"""
HASH_RE  = r"""\b[a-fA-F0-9]{40}\b"""  # sha1
ID_RE    = r"""\b[A-Za-z]{1,12}[A-Za-z0-9]{0,12}(?:[-_/][A-Za-z0-9]{1,24})+\b"""

DATE_YMD_RE = r"""\b\d{4}-\d{2}-\d{2}\b"""
DATE_DMY_RE = r"""\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"""
NUMBER_RE   = r"""\b\d+(?:[.,]\d+)?\b"""

BASE_WORD = r"[A-Za-zÀ-ÖØ-öø-ÿ]+"
ACRONYM   = r"[A-Z]{2,10}"
WORD = rf"{BASE_WORD}(?:(?:[’']{BASE_WORD})|(?:-(?:{BASE_WORD}|{ACRONYM})))*"

PUNCT_RE = r"""[“”"«»()\[\]{}…,:;.!?¿¡]"""
DASH_RE  = r"""[–—-]"""
BULLET_RE = r"""[•·]"""

TOKEN_RE = re.compile(
    rf"""
    (?:{URL_RE}) |
    (?:{EMAIL_RE}) |
    (?:{IPV4_RE}) |
    (?:{HASH_RE}) |
    (?:{ID_RE}) |
    (?:{DATE_YMD_RE}) |
    (?:{DATE_DMY_RE}) |
    (?:{NUMBER_RE}) |
    (?:{WORD}) |
    (?:{BULLET_RE}) |
    (?:{PUNCT_RE}) |
    (?:{DASH_RE}) |
    (?:\S)
    """,
    re.VERBOSE
)

TRAIL_PUNCT = set(".,;:!?)]}»")

@dataclass
class Tok:
    text: str
    start: int
    end: int

def _norm_apo(s: str) -> str:
    return (s or "").replace("’", "'")

def _has_letters(t: str) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", t or ""))

def _is_punct(t: str) -> bool:
    return bool(re.fullmatch(PUNCT_RE, t or "")) or bool(re.fullmatch(BULLET_RE, t or ""))

def _is_dash(t: str) -> bool:
    return bool(re.fullmatch(DASH_RE, t or ""))

def _is_number(t: str) -> bool:
    return bool(re.fullmatch(NUMBER_RE, t or "")) or bool(re.fullmatch(DATE_DMY_RE, t or "")) or bool(re.fullmatch(DATE_YMD_RE, t or ""))

def _is_acronym(t: str) -> bool:
    return bool(re.fullmatch(ACRONYM, t or ""))

def _is_capitalized_word(t: str) -> bool:
    return bool(re.fullmatch(r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ]+", t or ""))

def _is_hyphenated_word(t: str) -> bool:
    return "-" in (t or "") and bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]-[A-Za-zÀ-ÖØ-öø-ÿ]", t))

def _is_email(t: str) -> bool:
    return re.fullmatch(EMAIL_RE, t or "") is not None

def _is_url(t: str) -> bool:
    tl = (t or "").lower()
    return ("http://" in tl) or ("https://" in tl) or tl.startswith("www.")

def _is_ipv4(t: str) -> bool:
    return re.fullmatch(IPV4_RE, t or "") is not None

def _is_hash(t: str) -> bool:
    return re.fullmatch(HASH_RE, t or "") is not None

def _is_id(t: str) -> bool:
    return re.fullmatch(ID_RE, t or "") is not None

def _split_trailing_punct_if_needed(tk: Tok) -> List[Tok]:
    t = tk.text
    if not t or len(t) <= 1:
        return [tk]
    tl = t.lower()
    is_special = (
        ("http://" in tl) or ("https://" in tl) or tl.startswith("www.") or ("@" in t) or
        (_is_id(t)) or (_is_ipv4(t)) or (_is_hash(t))
    )
    if not is_special:
        return [tk]

    base = t
    end = tk.end
    trail = []
    while len(base) > 1 and base[-1] in TRAIL_PUNCT:
        trail.append(base[-1])
        base = base[:-1]
        end -= 1

    if not trail:
        return [tk]

    out = [Tok(base, tk.start, end)]
    cur = end
    for ch in reversed(trail):
        out.append(Tok(ch, cur, cur + 1))
        cur += 1
    return out

def tokenize_raw(text: str) -> List[Tok]:
    toks = [Tok(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text or "")]
    out: List[Tok] = []
    for tk in toks:
        out.extend(_split_trailing_punct_if_needed(tk))
    return out

ELISION_PREFIXES = {"d","l","j","t","m","s","n","c","qu","jusqu","lorsqu","puisqu"}
ELISION_EXCEPTIONS = {"aujourd'hui","aujourd’hui","quelqu'un","quelqu’un","presqu'île","presqu’île"}

def split_elisions(toks: List[Tok]) -> List[Tok]:
    out: List[Tok] = []
    for tk in toks:
        t = tk.text
        tl = _norm_apo(t).lower()
        if tl in ELISION_EXCEPTIONS:
            out.append(tk)
            continue

        m = re.match(r"^([A-Za-zÀ-ÖØ-öø-ÿ]+)(['’])([A-Za-zÀ-ÖØ-öø-ÿ].+)$", t)
        if not m:
            out.append(tk)
            continue

        pref = _norm_apo(m.group(1)).lower()
        if pref not in ELISION_PREFIXES:
            out.append(tk)
            continue

        cut = len(m.group(1)) + 1
        out.append(Tok(t[:cut], tk.start, tk.start + cut))     # ex: d'
        out.append(Tok(m.group(3), tk.start + cut, tk.end))    # ex: Air
    return out

def tokenize_with_spans(text: str) -> List[Tok]:
    return split_elisions(tokenize_raw(text))


# ============================================================
# 4) POS heuristique FR
# ============================================================
DET = {
    "le","la","les","un","une","des","du","de","au","aux","ce","cet","cette","ces",
    "mon","ma","mes","ton","ta","tes","son","sa","ses","notre","nos","votre","vos","leur","leurs"
}
PREP = {
    "à","a","de","dans","en","sur","sous","chez","vers","avec","sans","pour","par","entre","contre",
    "selon","depuis","pendant","outre","via"
}
CONJ = {"et","ou","mais","donc","or","ni","car","que"}
PRON = {
    "je","tu","il","elle","on","nous","vous","ils","elles",
    "me","te","se","lui","leur","y","en","ce","ça","cela","qui","quoi","dont","où"
}
PART = {"ne","pas","plus","jamais","rien","aucun","aucune","non"}
ADV = {"très","trop","bien","mal","ici","là","hier","aujourd'hui","aujourd’hui","demain","souvent","parfois","déjà","encore"}

CLITIC_POS_LEMMA = {
    "d'": ("IN",  "de"),
    "l'": ("DT",  "le"),
    "j'": ("PRP", "je"),
    "t'": ("PRP", "te"),
    "m'": ("PRP", "me"),
    "s'": ("PRP", "se"),
    "n'": ("RP",  "ne"),
    "c'": ("PRP", "ce"),
    "qu'":("CC",  "que"),
    "jusqu'": ("IN", "jusque"),
    "lorsqu'": ("CC", "lorsque"),
    "puisqu'": ("CC", "puisque"),
}

AUX_FORMS = {
    # avoir
    "ai","as","a","avons","avez","ont","avais","avait","avions","aviez","avaient",
    "aurai","auras","aura","aurons","aurez","auront",
    # être
    "suis","es","est","sommes","êtes","etes","sont","étais","etais","était","etait","étions","etions","étiez","etiez","étaient","etaient",
    "serai","seras","sera","serons","serez","seront",
}

NEG_MARKERS = {"ne","n'","pas","plus","jamais"}

def guess_pos(token: str, prev_tok: Optional[str], next_tok: Optional[str], prev_pos: Optional[str]) -> str:
    t = token or ""
    tn = _norm_apo(t)
    tl = tn.lower()

    if _is_punct(t) or _is_dash(t):
        return "PUNCT"
    if _is_number(t):
        return "CD"

    # Tokens tech: mieux en NNP qu'en NN (évite faux "NN fallback")
    if _is_url(t) or _is_email(t) or _is_ipv4(t) or _is_id(t) or _is_hash(t):
        return "NNP"

    if tl in CLITIC_POS_LEMMA:
        if tl == "l'" and next_tok:
            n2 = _norm_apo(next_tok).lower()
            if n2 in AUX_FORMS:
                return "PRP"
        return CLITIC_POS_LEMMA[tl][0]

    if tl in DET:
        return "DT"
    if tl in PREP:
        return "IN"
    if tl in CONJ:
        return "CC"
    if tl in PRON:
        return "PRP"
    if tl in PART:
        return "RP"
    if tl in ADV or tl.endswith("ment"):
        return "RB"

    if _is_acronym(t):
        return "NNP"

    if _is_hyphenated_word(t):
        parts = [p for p in t.split("-") if p]
        if any(_is_acronym(p) or _is_capitalized_word(p) for p in parts):
            return "NNP"
        if any(p.lower().endswith(("able","ible","ique","if","ive","eux","euse","aire","al","elle","iste")) for p in parts):
            return "JJ"
        return "NN"

    if _is_capitalized_word(t):
        return "NNP"

    if prev_tok:
        ptl = _norm_apo(prev_tok).lower()
        if ptl in AUX_FORMS:
            if re.search(r"(é|ée|ées|és|i|ie|ies|is|it|u|ue|ues|us)$", tl):
                return "VB"
        if prev_pos == "PRP" or ptl in NEG_MARKERS:
            if re.search(r"(e|es|ons|ez|ent|ais|ait|aient|era|erai|erons|erez|eront|irai|iras|ira|irons|irez|iront|ra|ront)$", tl):
                return "VB"

    if re.search(r"(er|ir|re|oir)$", tl):
        return "VB"

    if re.search(r"(able|ible|eux|euse|al|elle|ien|ienne|ique|if|ive|aux|aire|iste|ant|ante|ents|entes)$", tl):
        return "JJ"

    return "NN"


# ============================================================
# 5) Lemma FR (simplemma si dispo, sinon règles)
# ============================================================
IRREG_VERB = {
    "suis":"être","es":"être","est":"être","sommes":"être","êtes":"être","etes":"être","sont":"être",
    "ai":"avoir","as":"avoir","a":"avoir","avons":"avoir","avez":"avoir","ont":"avoir",
    "vais":"aller","vas":"aller","va":"aller","allons":"aller","allez":"aller","vont":"aller",
}

def lemma_fr(token: str, pos: str) -> str:
    t = token or ""
    tn = _norm_apo(t)
    tl = tn.lower()

    if _is_punct(t) or _is_dash(t):
        return "∅"
    if _is_number(t):
        return t

    if _is_url(t) or _is_email(t) or _is_ipv4(t) or _is_id(t) or _is_hash(t):
        return t

    if tl in CLITIC_POS_LEMMA:
        return CLITIC_POS_LEMMA[tl][1]

    if pos == "VB" and tl in IRREG_VERB:
        return IRREG_VERB[tl]

    if SIMPLEMMA_OK:
        try:
            return simplemma.lemmatize(tn, lang="fr")
        except Exception:
            pass

    if pos in ("NN", "NNP", "JJ"):
        if tl.endswith("aux") and len(tl) > 4:
            return tl[:-3] + "al"
        if tl.endswith(("s","x")) and len(tl) > 3:
            return tl[:-1]
        return tl

    if pos == "VB":
        if re.search(r"(ées|és|ée|é)$", tl):
            base = re.sub(r"(ées|és|ée|é)$", "", tl)
            return (base + "er") if base else tl
        return tl

    return tl


# ============================================================
# 6) NER: HF (si dispo) + fallback règles + DATE rules
# ============================================================
MOIS = {
    "janvier","février","fevrier","mars","avril","mai","juin","juillet","août","aout",
    "septembre","octobre","novembre","décembre","decembre"
}
JOURS = {"lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"}

ORG_FORMS = {"sarl","sas","sa","spa","eurl","inc","ltd","gmbh","llc"}
ORG_HINT  = {"université","ministère","groupe","société","compagnie","banque","association"}
TITLE_HINT = {"m.","mme","mlle","monsieur","madame","dr","docteur","prof","pr"}

def date_spans(text: str, toks: List[Tok]) -> List[Tuple[int,int,str]]:
    spans = []
    if not text:
        return spans

    for m in re.finditer(DATE_YMD_RE, text):
        spans.append((m.start(), m.end(), "DATE"))
    for m in re.finditer(DATE_DMY_RE, text):
        spans.append((m.start(), m.end(), "DATE"))
    for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", text):
        spans.append((m.start(), m.end(), "DATE"))

    i = 0
    while i < len(toks):
        w = _norm_apo(toks[i].text).lower()

        if re.fullmatch(r"\d{1,2}", toks[i].text) and i+1 < len(toks) and _norm_apo(toks[i+1].text).lower() in MOIS:
            j = i+2
            if j < len(toks) and re.fullmatch(r"(19\d{2}|20\d{2})", toks[j].text):
                j += 1
            spans.append((toks[i].start, toks[j-1].end, "DATE"))
            i = j
            continue

        if w in JOURS:
            j = i+1
            if j < len(toks) and re.fullmatch(r"\d{1,2}", toks[j].text):
                j += 1
            if j < len(toks) and _norm_apo(toks[j].text).lower() in MOIS:
                j += 1
            if j < len(toks) and re.fullmatch(r"(19\d{2}|20\d{2})", toks[j].text):
                j += 1
            if j > i+1:
                spans.append((toks[i].start, toks[j-1].end, "DATE"))
                i = j
                continue

        i += 1

    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    out = []
    last_end = -1
    for a,b,lab in spans:
        if a >= last_end:
            out.append((a,b,lab))
            last_end = b
    return out

def _norm_ner_label(raw: str) -> str:
    r = (raw or "").upper().strip()
    r = r.replace("I-", "").replace("B-", "")
    if r in ("PER", "PERSON"):
        return "PERS"
    if r in ("ORG", "LOC", "MISC", "PERS"):
        return r
    return r or "O"

def load_hf_ner(model_name: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.manual_seed(0)

    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        if "sentencepiece" in str(e).lower():
            raise RuntimeError("Ce modèle exige sentencepiece. Choisis un modèle BERT WordPiece (mBERT).") from e
        raise

    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    mdl.eval()

    ner = pipeline("ner", model=mdl, tokenizer=tok, device=-1, grouped_entities=True)
    return ner

def hf_spans(ner_pipe, text: str) -> List[Tuple[int,int,str]]:
    spans = []
    res = ner_pipe(text)
    for it in res:
        lab = it.get("entity_group") or it.get("entity") or ""
        lab = _norm_ner_label(lab)
        if lab in ("PERS","ORG","LOC","MISC"):
            a = int(it.get("start", -1))
            b = int(it.get("end", -1))
            if 0 <= a < b:
                spans.append((a,b,lab))

    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    out = []
    last_end = -1
    for a,b,lab in spans:
        if a >= last_end:
            out.append((a,b,lab))
            last_end = b
    return out

def apply_spans_to_tokens(toks: List[Tok], spans: List[Tuple[int,int,str]], labels: List[str]):
    for a,b,lab in spans:
        idxs = [i for i,t in enumerate(toks) if not (t.end <= a or t.start >= b)]
        if not idxs:
            continue
        if any(labels[i] != "O" for i in idxs):
            continue
        labels[idxs[0]] = f"B-{lab}"
        for i in idxs[1:]:
            labels[i] = f"I-{lab}"

def enforce_bio(labels: List[str]) -> List[str]:
    out = labels[:]
    for i in range(len(out)):
        if out[i].startswith("I-"):
            typ = out[i][2:]
            if i == 0 or out[i-1] == "O" or (out[i-1].startswith(("B-","I-")) and out[i-1][2:] != typ):
                out[i] = "B-" + typ
    return out

def ner_rules_spans(text: str, toks: List[Tok]) -> List[Tuple[int,int,str]]:
    spans: List[Tuple[int,int,str]] = []
    i = 0
    while i < len(toks):
        t = toks[i].text
        tl = _norm_apo(t).lower()

        # PERSON: titre + NNP+
        if tl.strip(".") in TITLE_HINT and i+1 < len(toks) and (_is_capitalized_word(toks[i+1].text) or _is_acronym(toks[i+1].text) or _is_hyphenated_word(toks[i+1].text)):
            a = toks[i].start
            j = i+1
            while j < len(toks) and (_is_capitalized_word(toks[j].text) or _is_acronym(toks[j].text) or _is_hyphenated_word(toks[j].text)):
                j += 1
            spans.append((a, toks[j-1].end, "PERS"))
            i = j
            continue

        # ORG: séquence de tokens (Capitalisés + petits connecteurs) + hints/forme juridique
        if _is_capitalized_word(t) or _is_acronym(t) or _is_hyphenated_word(t):
            a = toks[i].start
            j = i
            words = []
            while j < len(toks):
                w = toks[j].text
                wl = _norm_apo(w).lower()
                if _is_capitalized_word(w) or _is_acronym(w) or _is_hyphenated_word(w):
                    words.append(w)
                    j += 1
                    continue
                if wl in {"de","du","des","d'","d’","l'","l’","et","&"}:
                    words.append(w)
                    j += 1
                    continue
                if wl in ORG_FORMS or w.upper() in {"SARL","SAS","SA","SPA","EURL","INC","LTD","GMBH","LLC"}:
                    words.append(w)
                    j += 1
                    continue
                break

            low = " ".join(_norm_apo(x).lower() for x in words)
            if any(h in low for h in ORG_HINT) or any(f in low.split() for f in ORG_FORMS) or any(w.upper() in {"SARL","SAS","SA","SPA","EURL","INC","LTD","GMBH","LLC"} for w in words):
                spans.append((a, toks[j-1].end, "ORG"))
                i = j
                continue

        # LOC: préposition + NNP+
        if tl in PREP and i+1 < len(toks) and (_is_capitalized_word(toks[i+1].text) or _is_hyphenated_word(toks[i+1].text) or _is_acronym(toks[i+1].text)):
            a = toks[i+1].start
            j = i+1
            while j < len(toks) and (_is_capitalized_word(toks[j].text) or _is_acronym(toks[j].text) or _is_hyphenated_word(toks[j].text)):
                j += 1
            spans.append((a, toks[j-1].end, "LOC"))
            i = j
            continue

        i += 1

    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    out = []
    last_end = -1
    for a,b,lab in spans:
        if a >= last_end:
            out.append((a,b,lab))
            last_end = b
    return out


# ============================================================
# 7) POS boost par NER (ce que tu demandes)
# ============================================================
def improve_pos_with_ner(toks: List[Tok], pos: List[str], labels: List[str]) -> List[str]:
    """
    1) Si token est dans entité PERS/ORG/LOC/MISC => POS=NNP
    2) Si token est dans entité DATE => POS=CD si numérique, sinon NNP (mois/jour)
    3) Lexique intra-phrase: si un token alphabetique apparait en entité,
       on force aussi NNP quand il réapparait avec label O.
    """
    out = pos[:]

    # lexique intra-phrase (surface lower) venant du NER
    ent_lex = set()
    for i, lab in enumerate(labels):
        if lab == "O":
            continue
        typ = lab[2:] if len(lab) > 2 and lab[1] == "-" else lab
        if typ in ("PERS","ORG","LOC","MISC"):
            if _has_letters(toks[i].text):
                ent_lex.add(_norm_apo(toks[i].text).lower())

    for i, lab in enumerate(labels):
        t = toks[i].text
        if lab == "O":
            # re-application via lexique
            tl = _norm_apo(t).lower()
            if tl in ent_lex and _has_letters(t):
                out[i] = "NNP"
            continue

        typ = lab[2:] if lab[1] == "-" else lab
        if typ in ("PERS","ORG","LOC","MISC"):
            out[i] = "NNP"
        elif typ == "DATE":
            if _is_number(t):
                out[i] = "CD"
            else:
                out[i] = "NNP"

    return out


# ============================================================
# 8) Entities depuis BIO (join FR)
# ============================================================
NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", "…", ")", "]", "}", "»", "-", "—", "–", "/", "%"}
NO_SPACE_AFTER  = {"(", "[", "{", "«", "-", "—", "–", "/"}

def join_fr(tokens: List[str]) -> str:
    out = ""
    for tok in tokens:
        if not out:
            out = tok
            continue
        prev = out[-1]
        if tok in NO_SPACE_BEFORE:
            out += tok
        elif prev in NO_SPACE_AFTER:
            out += tok
        elif out.endswith("'") or out.endswith("’"):
            out += tok
        else:
            out += " " + tok
    return out

def entities_from_bio(toks: List[Tok], labels: List[str]) -> Dict[str, List[str]]:
    ents: Dict[str, List[str]] = {}
    i = 0
    while i < len(toks):
        lab = labels[i]
        if not lab.startswith("B-"):
            i += 1
            continue
        typ = lab[2:]
        j = i + 1
        parts = [toks[i].text]
        while j < len(toks) and labels[j] == f"I-{typ}":
            parts.append(toks[j].text)
            j += 1
        ents.setdefault(typ, []).append(join_fr(parts))
        i = j
    return ents


# ============================================================
# 9) Affichage
# ============================================================
def print_table(toks: List[Tok], pos: List[str], lem: List[str], max_rows: Optional[int] = None):
    if not toks:
        return
    w_token = max(10, min(32, max(len(t.text) for t in toks)))
    view = toks[:max_rows] if max_rows else toks
    for i, t in enumerate(view):
        print(f"{t.text:>{w_token}}  {pos[i]:<7} lemma={lem[i]}")


# ============================================================
# 10) Runner (comme EN): NER HF optionnel + fallback + POS boost
# ============================================================
def run_one(text: str, ner_pipe=None):
    print("=" * 90)
    print("INPUT:", text)

    toks = tokenize_with_spans(text)

    # POS initial + Lemma initial
    pos: List[str] = []
    lem: List[str] = []
    for i, tk in enumerate(toks):
        prev = toks[i-1].text if i > 0 else None
        nxt  = toks[i+1].text if i+1 < len(toks) else None
        prev_pos = pos[i-1] if i > 0 else None
        p = guess_pos(tk.text, prev, nxt, prev_pos)
        pos.append(p)
        lem.append(lemma_fr(tk.text, p))

    # NER
    labels = ["O"] * len(toks)
    ner_mode = "fallback (règles)"

    if ner_pipe is not None:
        try:
            spans = hf_spans(ner_pipe, text)
            apply_spans_to_tokens(toks, spans, labels)
            ner_mode = f"IA (transformers): {HF_MODEL_NAME}"
        except Exception as e:
            ner_mode = f"fallback (règles) - NER IA indisponible: {e}"
            labels = ["O"] * len(toks)

    # fallback règles si toujours vide
    if all(l == "O" for l in labels):
        r_sp = ner_rules_spans(text, toks)
        apply_spans_to_tokens(toks, r_sp, labels)

    # DATE rules (n’écrase pas une entité existante)
    d_sp = date_spans(text, toks)
    apply_spans_to_tokens(toks, d_sp, labels)

    labels = enforce_bio(labels)

    # POS boost via NER (ce que tu veux)
    pos2 = improve_pos_with_ner(toks, pos, labels)

    # Recompute lemma pour tokens impactés (simple et sûr)
    lem2: List[str] = []
    for i in range(len(toks)):
        if pos2[i] != pos[i]:
            lem2.append(lemma_fr(toks[i].text, pos2[i]))
        else:
            lem2.append(lem[i])

    # Print POS/Lemma
    print_table(toks, pos2, lem2, max_rows=None)

    # Print NER
    print()
    print(f"NER ({ner_mode}) (token, label):")
    print([(toks[i].text, labels[i]) for i in range(len(toks))])

    ents = entities_from_bio(toks, labels)
    print("Entities:")
    for k in sorted(ents.keys()):
        for v in ents[k]:
            print(f"  {k}: {v}")


# ============================================================
# 11) Split multi-sentences
# ============================================================
def split_input_into_sentences(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    dq = re.findall(r"\"([^\"]+)\"", s)
    gu = re.findall(r"«([^»]+)»", s)
    parts = [p.strip() for p in (dq + gu) if p.strip()]
    if parts:
        return parts
    parts = re.split(r"(?<=[\.\!\?])\s+", s)
    return [p.strip() for p in parts if p.strip()]


# ============================================================
# 12) Auto NER pipe (cached) + notebook-friendly entrypoints
# ============================================================
_FR_NER_PIPE = None

def get_ner_pipe():
    global _FR_NER_PIPE
    if _FR_NER_PIPE is not None:
        return _FR_NER_PIPE

    if not HF_OK:
        _FR_NER_PIPE = None
        return None

    try:
        _FR_NER_PIPE = load_hf_ner(HF_MODEL_NAME)
    except Exception as e:
        _FR_NER_PIPE = None
        print("[warn] (FR) NER IA not loaded, fallback rules. Cause:", e)

    return _FR_NER_PIPE

def run_one_auto(sentence: str):
    return run_one(sentence, ner_pipe=get_ner_pipe())

def run_from_previous_cell(data=None, max_sentences=None):
    try:
        from nb_utils import detect_lang, get_previous_cell_input, iter_sentences_from_input
    except Exception as e:
        raise ImportError(
            "nb_utils.py is required for run_from_previous_cell(). "
            "Put nb_utils.py in the same folder as frcode.py."
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
        if detect_lang(sent) != "fr":
            continue

        print()
        print("#" * 90)
        header = f"DOC={doc_name}"
        if page_idx is not None:
            header += f" | page={page_idx}"
        if sent_idx is not None:
            header += f" | sent={sent_idx}"
        header += " | lang=fr"
        print(header)
        print("#" * 90)

        out = run_one_auto(sent)
        results.append(out)

        count += 1
        if max_sentences is not None and count >= max_sentences:
            break

    return results


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="French POS/Lemma/NER (Notebook-friendly). "
                    "In Jupyter, use --from-previous-cell."
    )
    p.add_argument("--text", type=str, default="", help="Analyze a single sentence.")
    p.add_argument("--from-previous-cell", action="store_true",
                   help="Jupyter: use previous cell output as input (French only).")
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
