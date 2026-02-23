import os, re, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ----------------------------
# 0) Option deps local (sans venv)
# ----------------------------
LOCAL_DEPS_DIR = os.path.join(os.getcwd(), ".pylibs")
if os.path.isdir(LOCAL_DEPS_DIR) and LOCAL_DEPS_DIR not in sys.path:
    sys.path.insert(0, LOCAL_DEPS_DIR)

print("Python kernel:", sys.executable)
print("Local deps dir:", LOCAL_DEPS_DIR if os.path.isdir(LOCAL_DEPS_DIR) else "(absent)")

def print_install_help():
    py = sys.executable
    print("\n[install] Sans venv, en local dans .pylibs (utilise CE python):")
    print(f'  "{py}" -m pip install --upgrade --no-user-cfg --target "{LOCAL_DEPS_DIR}" nltk')
    print(f'  "{py}" -m pip install --upgrade --no-user-cfg --target "{LOCAL_DEPS_DIR}" torch transformers tokenizers')
    print("\nSi tu as: ERROR: Can not combine '--user' and '--target'")
    print("  PowerShell (désactive config pip qui force --user):")
    print('    $env:PIP_CONFIG_FILE="NUL"')
    print("    Remove-Item Env:PIP_USER -ErrorAction SilentlyContinue")
    print(f'    "{py}" -m pip install --upgrade --no-user-cfg --target "{LOCAL_DEPS_DIR}" nltk')

# ----------------------------
# 1) NLTK (optionnel): POS + WordNet lemma
# ----------------------------
NLTK_OK = False
_WORDNET_OK = False
_NLTK_IMPORT_ERR = None
WNL = None

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    NLTK_OK = True
except Exception as e:
    NLTK_OK = False
    _NLTK_IMPORT_ERR = e

def _ensure_nltk():
    if not NLTK_OK:
        return
    pkgs = [
        ("punkt", "tokenizers/punkt"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]
    for pkg, probe in pkgs:
        try:
            nltk.data.find(probe)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                print(f"[warn] NLTK download failed for {pkg}: {e}")

if NLTK_OK:
    _ensure_nltk()
    try:
        import nltk.corpus
        _ = nltk.corpus.wordnet.synsets("dog")
        _WORDNET_OK = True
    except Exception:
        _WORDNET_OK = False

if NLTK_OK and _WORDNET_OK:
    WNL = WordNetLemmatizer()

# ----------------------------
# 2) Transformers NER (optionnel) SANS sentencepiece
# ----------------------------
HF_OK = False
_HF_IMPORT_ERR = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    HF_OK = True
except Exception as e:
    HF_OK = False
    _HF_IMPORT_ERR = e

HF_MODEL_NAME = "dslim/bert-base-NER"  # BERT WordPiece => no sentencepiece

# ----------------------------
# 3) Tokenisation + décomposition contractions (avec offsets)
# ----------------------------
@dataclass
class Tok:
    text: str
    start: int
    end: int
    kind: str = "tok"          # "tok" | "punct" | "contr"
    hint_pos: Optional[str] = None
    hint_lemma: Optional[str] = None

# === PONCTUATION ET SYMBOLES: TOUJOURS tag PUNCT + lemma ∅ ===
# Standard: . , ; : ! ? ( ) [ ] { } < > " '
# Guillemets: " ' " '
# Tirets: — – -
# Symboles: … @ / \ | & = + * ^ ~ ` %
PUNCT_RE = re.compile(r"^[\"\"\"'''()\[\]{}…,:;.!?@/|&=+*^~`%\\\-–—]$")

def _is_punct(t: str) -> bool:
    return bool(PUNCT_RE.match(t))

def _is_number(t: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)?", t)) or bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)) or bool(re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", t))

MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

# NOTE: inclut aussi les composés digit-hyphen: 20-year-old
TOKEN_RE = re.compile(
    rf"""
    (?:\d{{4}}-\d{{2}}-\d{{2}}) |                          # 2026-02-12
    (?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}) |              # 02/12/2026
    (?:{MONTHS}\s+\d{{1,2}}(?:,\s*\d{{2,4}})?) |           # Feb 12, 2026
    (?:\d{{1,2}}\s+{MONTHS}(?:\s+\d{{2,4}})?) |            # 12 Feb 2026
    (?:\d+(?:-[A-Za-z]+)+) |                               # 20-year-old
    (?:[A-Za-z]\.){{2,}} |                                 # U.S. / Ph.D.
    (?:\d+(?:[.,]\d+)?) |                                  # numbers
    (?:[A-Za-z]+(?:[’'][A-Za-z]+)+) |                      # words with apostrophes (candidates contractions)
    (?:[A-Za-z]+(?:-[A-Za-z]+)+) |                         # hyphen compounds
    (?:[A-Za-z]+) |                                        # plain words
    (?:[“”"‘’'()\[\]{{}}…,:;.!?]) |                        # punctuation
    (?:\S)                                                 # fallback 1 char
    """,
    re.VERBOSE,
)

def _lower(s: str) -> str:
    return (s or "").lower()

def _has_apostrophe(s: str) -> bool:
    return ("'" in s) or ("’" in s)

def _norm_apos(s: str) -> str:
    return (s or "").replace("’", "'")

# special irregular contractions
IRREG_BASE = {
    "can't": "can",
    "won't": "will",
    "shan't": "shall",
    "ain't": "be",
}

# suffix -> expansion (deterministic)
SUFFIX_EXPAND = {
    "n't": ("not", "RB", "not"),
    "'ve": ("have", "VB", "have"),
    "'re": ("are",  "VB", "be"),
    "'ll": ("will", "VB", "will"),
    "'m":  ("am",   "VB", "be"),
    "'d":  ("would","VB", "would"),   # deterministe: would (pas had)
    "'s":  ("__S__", None, None),      # résolu ensuite: POS ou is/has
}

def _looks_like_word(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", s)) or bool(re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)+", s)) or bool(re.fullmatch(r"\d+(?:-[A-Za-z]+)+", s))

def _split_plural_possessive(tok: str, abs_start: int) -> Optional[List[Tok]]:
    # Joneses'  -> Joneses + '
    t = _norm_apos(tok)
    if re.fullmatch(r"[A-Za-z]+s'", t) and len(t) >= 3:
        base_len = len(t) - 1
        base = tok[:-1]  # keep original char
        return [
            Tok(base, abs_start, abs_start + base_len, kind="tok"),
            Tok("'", abs_start + base_len, abs_start + base_len + 1, kind="contr", hint_pos="POS", hint_lemma="∅"),
        ]
    return None

def split_contractions(tok: str, abs_start: int) -> List[Tok]:
    """
    Décompose les contractions, y compris chaînes: shouldn't've.
    Ne casse pas les noms type O'Reilly (suffix non reconnu).
    """
    # 1) plural possessive
    pp = _split_plural_possessive(tok, abs_start)
    if pp is not None:
        return pp

    raw = tok
    t = _norm_apos(raw)

    # si pas apostrophe, rien
    if "'" not in t:
        return [Tok(raw, abs_start, abs_start + len(raw), kind="tok")]

    # si suffix inconnu (ex: O'Reilly), on garde
    # on décide de splitter seulement si on peut matcher un suffix de contraction à la fin.
    # (sauf pour chaînes où on va enlever plusieurs suffixes)
    cur_text = t
    cur_end = abs_start + len(raw)

    suffix_toks_rev: List[Tok] = []

    # boucle "peel suffixes"
    while True:
        matched = False
        for suf in ["n't", "'ve", "'re", "'ll", "'m", "'d", "'s"]:
            if cur_text.endswith(suf):
                matched = True
                # span suffix dans le token original
                suf_len = len(suf)
                suf_start = cur_end - suf_len
                suf_end = cur_end

                exp_txt, exp_pos, exp_lem = SUFFIX_EXPAND[suf]

                suffix_toks_rev.append(
                    Tok(exp_txt, suf_start, suf_end, kind="contr", hint_pos=exp_pos, hint_lemma=exp_lem)
                )

                # strip suffix
                cur_end -= suf_len
                cur_text = cur_text[:-suf_len]
                break
        if not matched:
            break

    # si aucun suffix reconnu, pas contraction
    if not suffix_toks_rev:
        return [Tok(raw, abs_start, abs_start + len(raw), kind="tok")]

    base_raw = raw[: (cur_end - abs_start)]
    base_norm = _norm_apos(base_raw)

    # cas irréguliers (can't/won't/...)
    bn_low = base_norm.lower() + ("n't" if t.endswith("n't") and not base_norm.endswith("n") else "")
    # simplifie: on check l’original complet
    full_low = _norm_apos(raw).lower()
    if full_low in IRREG_BASE:
        # base span = partie avant "n't" (souvent "ca"/"wo"), mais texte = can/will
        base_text = IRREG_BASE[full_low]
        base_tok = Tok(base_text, abs_start, cur_end, kind="contr", hint_pos="VB", hint_lemma=base_text)
    else:
        base_tok = Tok(base_raw if base_raw else raw, abs_start, cur_end, kind="tok")

    # remettre dans l’ordre: base + suffixes (reverse)
    suffix_toks = list(reversed(suffix_toks_rev))
    out = [base_tok] + suffix_toks

    return out

def tokenize_with_spans(text: str) -> List[Tok]:
    toks: List[Tok] = []
    for m in TOKEN_RE.finditer(text or ""):
        s = m.group(0)
        a, b = m.start(), m.end()

        if _is_punct(s):
            toks.append(Tok(s, a, b, kind="punct", hint_pos="PUNCT", hint_lemma="∅"))
            continue

        if _has_apostrophe(s) and _looks_like_word(s):
            toks.extend(split_contractions(s, a))
        else:
            toks.append(Tok(s, a, b, kind="tok"))
    return toks

# ----------------------------
# 4) POS: NLTK si dispo, sinon heuristiques + mapping tags simplifiés
# ----------------------------
DET = {"the","a","an","this","that","these","those","my","your","his","her","its","our","their"}
PREP = {"in","on","at","to","from","of","for","with","without","by","between","among","into","onto","over","under","about","across","through","during","before","after","since"}
CONJ = {"and","or","but","so","yet","nor"}
PRON = {"i","you","he","she","it","we","they","me","him","her","us","them","who","whom","whose","which","that"}
ADV = {"very","too","well","badly","here","there","yesterday","today","tomorrow","often","sometimes","already","still","also"}

def guess_pos_heur(token: str, prev_tok: Optional[str], next_tok: Optional[str]) -> str:
    tl = _lower(token)

    if _is_punct(token):
        return "PUNCT"
    if _is_number(token):
        return "CD"
    if tl in DET:
        return "DT"
    if tl in PREP:
        return "IN"
    if tl in CONJ:
        return "CC"
    if tl in PRON:
        return "PRP"
    if tl in ADV:
        return "RB"
    if tl in {"not"}:
        return "RB"
    if tl in {"would","will","have","has","had","am","are","is","be","been","being"}:
        return "VB"

    # capitalized -> NNP (sauf articles)
    if re.fullmatch(r"[A-Z][a-z]+(?:['’][A-Za-z]+)?", token) and tl not in DET and tl not in PRON:
        return "NNP"

    # verb-ish
    if re.search(r"(ing|ed)$", tl) or re.search(r"(ize|ise|ate|fy)$", tl):
        return "VB"

    # adjective-ish
    if re.search(r"(able|ible|ous|ive|al|ic|ish|less|ful)$", tl):
        return "JJ"

    # noun plural-ish
    if tl.endswith("s") and len(tl) > 3:
        return "NN"

    return "NN"

def map_penn_to_simple(p: str) -> str:
    if not p:
        return "NN"
    pu = p.upper()

    if pu in {".", ",", ":", "``", "''", "-LRB-", "-RRB-"}:
        return "PUNCT"

    if pu.startswith("NNP") or pu == "NNPS":
        return "NNP"
    if pu.startswith("NN"):
        return "NN"
    if pu.startswith("VB") or pu == "MD":
        return "VB"
    if pu.startswith("JJ"):
        return "JJ"
    if pu.startswith("RB"):
        return "RB"
    if pu in {"IN", "TO"}:
        return "IN"
    if pu in {"DT", "PDT"}:
        return "DT"
    if pu in {"PRP", "PRP$", "WP", "WP$"}:
        return "PRP"
    if pu == "WDT":
        return "WDT"
    if pu == "CC":
        return "CC"
    if pu == "CD":
        return "CD"
    if pu == "RP":
        return "RP"
    if pu == "POS":
        return "POS"

    return "NN"

def pos_tag_simple(tokens: List[str]) -> Tuple[List[str], str]:
    if NLTK_OK:
        try:
            tagged = nltk.pos_tag(tokens)
            mapped = [map_penn_to_simple(p) for _, p in tagged]
            return mapped, "NLTK (mapped->simple)"
        except Exception as e:
            pass
    # fallback heuristics
    pos = []
    for i, t in enumerate(tokens):
        prev = tokens[i-1] if i > 0 else None
        nxt = tokens[i+1] if i+1 < len(tokens) else None
        pos.append(guess_pos_heur(t, prev, nxt))
    return pos, "heuristics"

# ----------------------------
# 5) Résolution de __S__ (copule vs possessif)
# ----------------------------
PRON_SUBJ = {"i","you","he","she","it","we","they","who","that","there","here","what","where","when","how"}
def looks_like_nounish(tok: str) -> bool:
    if not tok:
        return False
    if _is_number(tok):
        return True
    if re.fullmatch(r"[A-Z][a-z]+", tok):
        return True
    if re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)+", tok):
        return True
    if re.fullmatch(r"[A-Za-z]+", tok) and tok.lower() not in DET and tok.lower() not in PREP and tok.lower() not in CONJ:
        return True
    return False

def resolve_s_contractions(toks: List[Tok]) -> None:
    """
    Modifie en place:
      - __S__ -> is (copule) OU 's (POS)
    """
    for i, tk in enumerate(toks):
        if tk.text != "__S__":
            continue
        prev = toks[i-1].text if i > 0 else ""
        nxt = toks[i+1].text if i+1 < len(toks) else ""
        prev_l = prev.lower()

        # Si précédent est pronom sujet -> "is"
        if prev_l in PRON_SUBJ:
            tk.text = "is"
            tk.hint_pos = "VB"
            tk.hint_lemma = "be"
            continue

        # Si suivant ressemble à un nom/adjectif (ex: John's car / John's big car) -> possessif
        # Heuristique: si suivant est "nounish" -> POS
        if looks_like_nounish(nxt) or nxt.lower() in DET:
            tk.text = "'s"
            tk.hint_pos = "POS"
            tk.hint_lemma = "∅"
            continue

        # Sinon -> copule
        tk.text = "is"
        tk.hint_pos = "VB"
        tk.hint_lemma = "be"

# ----------------------------
# 6) Lemma: WordNet si dispo, sinon règles + hints
# ----------------------------
def _penn_to_wordnet_simple(pos: str) -> str:
    if not pos:
        return "n"
    p = pos.upper()
    if p.startswith("VB"):
        return "v"
    if p.startswith("JJ"):
        return "a"
    if p.startswith("RB"):
        return "r"
    return "n"

def lemma_en(token: str, pos: str) -> str:
    if _is_punct(token):
        return "∅"
    if _is_number(token):
        return token

    tl = token.lower()

    # possessive marker tokens
    if token in {"'s", "'"}:
        return "∅"

    # WordNet lemmatizer
    if WNL is not None:
        try:
            return WNL.lemmatize(tl, _penn_to_wordnet_simple(pos))
        except Exception:
            pass

    # fallback rules
    if pos == "NN" and tl.endswith("ies") and len(tl) > 4:
        return tl[:-3] + "y"
    if pos == "NN" and tl.endswith("s") and not tl.endswith("ss") and len(tl) > 3:
        return tl[:-1]

    if pos == "VB":
        if tl.endswith("ing") and len(tl) > 5:
            base = tl[:-3]
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if tl.endswith("ed") and len(tl) > 4:
            base = tl[:-2]
            return base
        return tl

    return tl

# ----------------------------
# 7) Fix "Buffalo" (verbe) déterministe
# ----------------------------
def fix_buffalo_pos(tokens: List[str], pos: List[str], lem: List[str]) -> None:
    idxs = [i for i,t in enumerate(tokens) if t.lower() == "buffalo"]
    if len(idxs) < 4:
        return  # on évite de sur-corriger les phrases normales

    for i in range(1, len(tokens)-1):
        if tokens[i].lower() != "buffalo":
            continue
        # "slot verbe" : entouré de noms/proper nouns
        if pos[i] in {"NN","NNP"} and pos[i-1] in {"NN","NNP"} and pos[i+1] in {"NN","NNP"}:
            pos[i] = "VB"
            lem[i] = "buffalo"

# ----------------------------
# 8) DATE spans (règles) + NER 30% IA + fixes bruit Buffalo
# ----------------------------
def date_spans(text: str) -> List[Tuple[int,int,str]]:
    spans = []
    if not text:
        return spans

    for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        spans.append((m.start(), m.end(), "DATE"))
    for m in re.finditer(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
        spans.append((m.start(), m.end(), "DATE"))
    for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", text):
        spans.append((m.start(), m.end(), "DATE"))
    for m in re.finditer(rf"\b{MONTHS}\s+\d{{1,2}}(?:,\s*\d{{2,4}})?\b", text):
        spans.append((m.start(), m.end(), "DATE"))
    for m in re.finditer(rf"\b\d{{1,2}}\s+{MONTHS}(?:\s+\d{{2,4}})?\b", text):
        spans.append((m.start(), m.end(), "DATE"))

    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    out = []
    last_end = -1
    for a,b,lab in spans:
        if a >= last_end:
            out.append((a,b,lab))
            last_end = b
    return out

def load_hf_ner(model_name: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    mdl.eval()

    ner = pipeline(
        "ner",
        model=mdl,
        tokenizer=tok,
        device=-1,
        grouped_entities=True
    )
    return ner

def _norm_ner_label(raw: str) -> str:
    r = (raw or "").upper().strip()
    r = r.replace("I-", "").replace("B-", "")
    if r in ("PER", "PERSON"):
        return "PERS"
    if r in ("ORG", "LOC", "MISC", "PERS"):
        return r
    return r or "O"

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

def fix_buffalo_ner(tokens: List[str], labels: List[str]) -> List[str]:
    """
    Débruite:
      - Buffalo (majuscule) taggué MISC -> LOC
      - buffalo (minuscule) taggué MISC -> O
    """
    out = labels[:]
    for i, tok in enumerate(tokens):
        if tok == "Buffalo" and out[i].endswith("MISC"):
            out[i] = out[i].replace("MISC", "LOC")
        if tok == "buffalo" and out[i].endswith("MISC"):
            out[i] = "O"
    return enforce_bio(out)

# ----------------------------
# 9) Entities depuis BIO
# ----------------------------
def _join_entity_tokens(parts: List[str]) -> str:
    s = " ".join(parts)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()

def entities_from_bio(tokens: List[str], labels: List[str]) -> Dict[str, List[str]]:
    ents: Dict[str, List[str]] = {}
    i = 0
    while i < len(tokens):
        lab = labels[i]
        if not lab.startswith("B-"):
            i += 1
            continue
        typ = lab[2:]
        j = i + 1
        parts = [tokens[i]]
        while j < len(tokens) and labels[j] == f"I-{typ}":
            parts.append(tokens[j])
            j += 1
        val = _join_entity_tokens([p for p in parts if p not in {"'"}])
        if val:
            ents.setdefault(typ, []).append(val)
        i = j
    return ents

# ----------------------------
# 10) Affichage "style arabe"
# ----------------------------
def print_table(tokens: List[str], pos: List[str], lem: List[str], max_rows: Optional[int] = None):
    w_token = max(10, min(30, max(len(t) for t in tokens) if tokens else 10))
    rows = range(len(tokens)) if max_rows is None else range(min(len(tokens), max_rows))
    for i in rows:
        print(f"{tokens[i]:>{w_token}}  {pos[i]:<7} lemma={lem[i]}")

# ----------------------------
# 11) Fallback NER (si HF indispo): capitalized sequences + dates
# ----------------------------
def fallback_ner_labels(tokens: List[str], pos: List[str]) -> List[str]:
    labels = ["O"] * len(tokens)

    # simple: suite de mots capitalisés -> MISC, après prépositions -> LOC
    preps = {"in","at","from","to","near","around","within","outside","inside","of"}
    i = 0
    while i < len(tokens):
        if re.fullmatch(r"[A-Z][a-z]+", tokens[i]):
            j = i + 1
            while j < len(tokens) and re.fullmatch(r"[A-Z][a-z]+", tokens[j]):
                j += 1
            typ = "MISC"
            if i > 0 and tokens[i-1].lower() in preps:
                typ = "LOC"
            labels[i] = f"B-{typ}"
            for k in range(i+1, j):
                labels[k] = f"I-{typ}"
            i = j
            continue
        i += 1

    return enforce_bio(labels)

# ----------------------------
# 12) Runner
# ----------------------------
def run_one(text: str, ner_pipe=None):
    print("=" * 90)
    print("INPUT:", text)

    toks = tokenize_with_spans(text)
    resolve_s_contractions(toks)

    tokens = [t.text for t in toks]

    # POS
    pos, pos_mode = pos_tag_simple(tokens)

    # appliquer hints de tokenisation (not/have/would/POS...)
    for i, tk in enumerate(toks):
        if tk.hint_pos is not None:
            pos[i] = tk.hint_pos

    # Lemma
    lem = []
    lemma_mode = "fallback rules"
    if WNL is not None:
        lemma_mode = "NLTK WordNetLemmatizer"
    for i, tk in enumerate(toks):
        if tk.hint_lemma is not None:
            lem.append(tk.hint_lemma)
        else:
            lem.append(lemma_en(tk.text, pos[i]))

    # Fix Buffalo POS
    fix_buffalo_pos(tokens, pos, lem)

    # Print POS/lemma table
    print_table(tokens, pos, lem, max_rows=None)

    # NER
    labels = ["O"] * len(toks)
    ner_mode = "fallback (rules)"

    if ner_pipe is not None:
        try:
            spans = hf_spans(ner_pipe, text)
            apply_spans_to_tokens(toks, spans, labels)
            ner_mode = f"IA (transformers): {HF_MODEL_NAME}"
        except Exception as e:
            ner_mode = f"fallback (rules) - NER 30% IA indisponible: {e}"
            labels = fallback_ner_labels(tokens, pos)
    else:
        labels = fallback_ner_labels(tokens, pos)

    # DATE rules (n’écrase pas une entité existante)
    d_sp = date_spans(text)
    apply_spans_to_tokens(toks, d_sp, labels)
    labels = enforce_bio(labels)

    # Fix bruit Buffalo
    labels = fix_buffalo_ner(tokens, labels)

    # Print NER output
    print()
    print(f"NER ({ner_mode}) (token, label):")
    print(list(zip(tokens, labels)))

    ents = entities_from_bio(tokens, labels)
    print("Entities:")
    for k in sorted(ents.keys()):
        for v in ents[k]:
            print(f"  {k}: {v}")

    # Audit (heuristique, pas une mesure de justesse)
    total = max(1, len(tokens))
    nn_like = sum(1 for p in pos if p in ("NN", "NNP"))
    contr = sum(1 for t in toks if t.kind == "contr" or t.text in {"not","have","would","is","'s","'"} )
    print()
    print("Audit (heuristique, pas une mesure de justesse):")
    print(f"  tokens total     : {len(tokens)}")
    print(f"  POS mode         : {pos_mode}")
    print(f"  lemma mode       : {lemma_mode}")
    print(f"  NN/NNP ratio     : {nn_like}/{len(pos)} = {nn_like/total*100:.2f} %")
    print(f"  contraction tokens (approx): {contr} ({contr/total*100:.2f} %)")

def split_input_into_sentences(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []

    # Split sur les points, points d'exclamation, points d'interrogation uniquement
    parts = re.split(r"[\.\!\?]+", s)
    return [p.strip() for p in parts if p.strip()]

# ----------------------------
# Auto NER pipe (cached) + notebook-friendly entrypoints
# ----------------------------
_EN_NER_PIPE = None

def get_ner_pipe():
    global _EN_NER_PIPE
    if _EN_NER_PIPE is not None:
        return _EN_NER_PIPE

    if not HF_OK:
        _EN_NER_PIPE = None
        return None

    try:
        _EN_NER_PIPE = load_hf_ner(HF_MODEL_NAME)
    except Exception as e:
        _EN_NER_PIPE = None
        print("[warn] (EN) NER 30% IA not loaded, fallback rules. Cause:", e)

    return _EN_NER_PIPE

def run_one_auto(sentence: str):
    """Run EN pipeline on one sentence (loads NER lazily if available)."""
    return run_one(sentence, ner_pipe=get_ner_pipe())

def run_from_previous_cell(data=None, max_sentences=None):
    """
    Jupyter-only helper: fetch previous cell output and run EN pipeline
    only on sentences detected as English.
    """
    try:
        from nb_utils import detect_lang, get_previous_cell_input, iter_sentences_from_input
    except Exception as e:
        raise ImportError(
            "nb_utils.py is required for run_from_previous_cell(). "
            "Put nb_utils.py in the same folder as engcode.py."
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
        if detect_lang(sent) != "en":
            continue

        print()
        print("#" * 90)
        header = f"DOC={doc_name}"
        if page_idx is not None:
            header += f" | page={page_idx}"
        if sent_idx is not None:
            header += f" | sent={sent_idx}"
        header += " | lang=en"
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
        description="English POS/Lemma/NER (Notebook-friendly). "
                    "In Jupyter, use --from-previous-cell."
    )
    p.add_argument("--text", type=str, default="", help="Analyze a single sentence.")
    p.add_argument("--from-previous-cell", action="store_true",
                   help="Jupyter: use previous cell output as input (English only).")
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
