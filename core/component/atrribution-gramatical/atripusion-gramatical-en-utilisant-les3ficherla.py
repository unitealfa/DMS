# =========================
# 1) Chemin vers tes .py
# =========================
import sys, types, re, importlib, os, unicodedata
from collections import Counter
from pathlib import Path

# Utilise le dossier du composant pour rester portable
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Mode offline : forcer ici (MANUAL_OFFLINE=True) ou via l'env LANG_PIPE_OFFLINE=1
MANUAL_OFFLINE = False
OFFLINE = MANUAL_OFFLINE or (os.environ.get("LANG_PIPE_OFFLINE", "0").lower() in {"1", "true", "yes", "on"})
if OFFLINE:
    print("[info] Mode offline: NER désactivé (eng/fr).")

# =========================
# 2) Petit "nb_utils" en mémoire (pas besoin de créer nb_utils.py)
#    -> utilisé par run_from_previous_cell() dans tes scripts
# =========================
nb_utils = types.ModuleType("nb_utils")

_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", flags=re.UNICODE)
_FR_HINT = {"le","la","les","des","une","un","est","avec","pour","dans","sur","facture","date","total","tva","montant"}
_EN_HINT = {"the","and","to","of","in","is","for","with","invoice","date","total","vat","amount"}

def _is_punct_like_token(token: object) -> bool:
    txt = str(token or "").strip()
    if not txt:
        return False
    if txt in {"_", "∅"}:
        return True
    for ch in txt:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P"):
            continue
        if ch in {"∅"}:
            continue
        return False
    return True

def _normalize_token_fields(token: object, pos_value: object, lemma_value: object):
    tok = str(token or "")
    pos_txt = str(pos_value or "")
    lemma_txt = str(lemma_value or "")
    punct_from_token = _is_punct_like_token(tok)
    punct_from_lemma = lemma_txt.strip() in {"_", "∅"}

    if punct_from_token or punct_from_lemma:
        pos_txt = "PUNCT"
        if not lemma_txt.strip() or lemma_txt.strip() in {"_", "∅"}:
            lemma_txt = tok.strip() or "_"
    return tok, pos_txt, lemma_txt

def detect_lang(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "en"

    if _AR_RE.search(t):
        return "ar"

    words = [w.lower() for w in _WORD_RE.findall(t[:8000])]
    if not words:
        return "en"

    # listes un peu plus riches pour éviter les égalités débiles
    fr_hint = _FR_HINT | {
        "de", "du", "au", "aux", "et", "ou", "mais",
        "ce", "cet", "cette", "ces",
        "il", "elle", "nous", "vous", "ils", "elles",
        "son", "sa", "ses", "dans", "sur", "avec", "pour"
    }
    en_hint = _EN_HINT | {
        "a", "an", "this", "that", "these", "those",
        "it", "its", "he", "she", "we", "they",
        "as", "also", "from", "on", "by", "into", "over", "under"
    }

    fr_score = sum(1 for w in words if w in fr_hint)
    en_score = sum(1 for w in words if w in en_hint)

    # accent = gros indice FR
    if re.search(r"[éèêàùçôîï]", t.lower()):
        fr_score += 2

    # égalité => ne plus favoriser le FR
    if fr_score > en_score:
        return "fr"
    return "en"

def get_previous_cell_input():
    g = globals()
    for k in ("selected", "TOK_DOCS", "FINAL_DOCS", "DOCS", "TEXT_DOCS", "_"):
        if k not in g or g[k] is None:
            continue
        val = g[k]
        if isinstance(val, list) and not val:
            continue
        return val
    return None

def iter_sentences_from_input(data):
    """
    Yield: (doc_name, page_idx, sent_idx, sent_text)
    Supporte: TOK_DOCS/selected (pages->sentences_layout), FINAL_DOCS (list[{text}]), etc.
    """
    if data is None:
        return

    # Cas 1: liste de docs avec pages (TOK_DOCS / selected)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "pages" in data[0]:
        for d_i, doc in enumerate(data):
            doc_name = doc.get("filename") or doc.get("doc_id") or f"doc#{d_i}"
            pages = doc.get("pages") or []
            for p_i, pg in enumerate(pages):
                page_idx = pg.get("page_index", pg.get("page", p_i+1))
                sent_items = pg.get("sentences_layout") or pg.get("sentences") or pg.get("chunks") or []
                for s_i, s in enumerate(sent_items):
                    if isinstance(s, dict):
                        if s.get("is_sentence") is False:
                            continue
                        sent = s.get("text") or ""
                    else:
                        sent = str(s)
                    yield doc_name, page_idx, s_i, sent
        return

    # Cas 2: FINAL_DOCS : list[{text, filename?}]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
        for i, d in enumerate(data):
            doc_name = d.get("filename") or d.get("doc_id") or f"doc#{i}"
            yield doc_name, None, None, d.get("text") or ""
        return

    # Cas 3: dict {text:...}
    if isinstance(data, dict) and "text" in data:
        doc_name = data.get("filename") or data.get("doc_id") or "doc"
        yield doc_name, None, None, data.get("text") or ""
        return

    # Cas 4: string direct
    if isinstance(data, str):
        yield "text", None, None, data
        return

    raise TypeError(f"Format d'entrée non supporté: {type(data)}")

nb_utils.detect_lang = detect_lang
nb_utils.get_previous_cell_input = get_previous_cell_input
nb_utils.iter_sentences_from_input = iter_sentences_from_input
sys.modules["nb_utils"] = nb_utils  # rend "import nb_utils" possible

# =========================
# 3) Import + reload tes 3 modules
# =========================
import engcode
import frcode

# arabcode peut échouer si camel_tools n'est pas installé => on skip proprement
try:
    import arabcode
    HAVE_AR = True
except Exception as e:
    HAVE_AR = False
    print("[warn] arabcode.py non chargé (dépendances manquantes ?). Détail:", e)

importlib.reload(engcode)
importlib.reload(frcode)
if HAVE_AR:
    importlib.reload(arabcode)

# Si offline, désactiver explicitement les pipelines 30% IA ENG/FR (fallback règles uniquement)
if OFFLINE:
    if hasattr(engcode, "HF_OK"):
        engcode.HF_OK = False
    if hasattr(engcode, "_EN_NER_PIPE"):
        engcode._EN_NER_PIPE = None
    if hasattr(frcode, "HF_OK"):
        frcode.HF_OK = False
    if hasattr(frcode, "_FR_NER_PIPE"):
        frcode._FR_NER_PIPE = None

# =========================
# 4) Exécution: chaque script filtre sa langue et print son output
# =========================
data = get_previous_cell_input()
if data is None:
    raise RuntimeError("Je ne trouve pas de données d'entrée. Assure-toi que la cellule précédente crée 'selected' (ou FINAL_DOCS / TOK_DOCS).")

MAX_SENTENCES_PER_LANG = None  # ex: 30 pour debug, ou None pour tout

all_sentences = []
for doc_name, page_idx, sent_idx, sent in iter_sentences_from_input(data):
    sent = (sent or "").strip()
    if sent:
        all_sentences.append((doc_name, page_idx, sent_idx, sent, detect_lang(sent)))

has_en = any(lang == "en" for _, _, _, _, lang in all_sentences)
has_fr = any(lang == "fr" for _, _, _, _, lang in all_sentences)
has_ar = any(lang == "ar" for _, _, _, _, lang in all_sentences)

EN_RESULTS = []
FR_RESULTS = []
AR_RESULTS = []

if has_en:
    print("\n" + "="*120)
    print("RUN EN (engcode.py)")
    print("="*120)
    EN_RESULTS = engcode.run_from_previous_cell(data=data, max_sentences=MAX_SENTENCES_PER_LANG) or []

if has_fr:
    print("\n" + "="*120)
    print("RUN FR (frcode.py)")
    print("="*120)
    FR_RESULTS = frcode.run_from_previous_cell(data=data, max_sentences=MAX_SENTENCES_PER_LANG) or []

if HAVE_AR and has_ar:
    print("\n" + "="*120)
    print("RUN AR (arabcode.py)")
    print("="*120)
    AR_RESULTS = arabcode.run_from_previous_cell(data=data, max_sentences=MAX_SENTENCES_PER_LANG) or []

NLP_ANALYSES = [x for x in (EN_RESULTS + FR_RESULTS + AR_RESULTS) if isinstance(x, dict)]
NLP_SENTENCES = []
NLP_ENTITIES = []
NLP_TOKENS = []
_lang_counter = Counter()

for row in NLP_ANALYSES:
    lang = str(row.get("lang") or "unknown")
    _lang_counter[lang] += 1
    sent_text = str(row.get("text") or "")
    filename = row.get("filename") or row.get("doc")
    page_index = row.get("page_index")
    sent_index = row.get("sent_index")

    NLP_SENTENCES.append(
        {
            "filename": filename,
            "page_index": page_index,
            "sent_index": sent_index,
            "lang": lang,
            "text": sent_text,
        }
    )

    entities = row.get("entities")
    if isinstance(entities, dict):
        if "flat" in entities and isinstance(entities.get("flat"), list):
            for ent in entities.get("flat") or []:
                if not isinstance(ent, dict):
                    continue
                NLP_ENTITIES.append(
                    {
                        "filename": filename,
                        "page_index": page_index,
                        "sent_index": sent_index,
                        "lang": lang,
                        "type": ent.get("type"),
                        "text": ent.get("text"),
                    }
                )
        else:
            for etype, values in entities.items():
                if not isinstance(values, list):
                    continue
                for val in values:
                    NLP_ENTITIES.append(
                        {
                            "filename": filename,
                            "page_index": page_index,
                            "sent_index": sent_index,
                            "lang": lang,
                            "type": etype,
                            "text": val,
                        }
                    )

    tokens = row.get("tokens") or []
    pos = row.get("pos") or []
    lemmas = row.get("lemmas") or []
    ner = row.get("ner_labels") or []
    size = len(tokens)
    if isinstance(ner, list):
        ner_view = ner
    else:
        ner_view = []

    for i in range(size):
        tok = tokens[i]
        base_pos = pos[i] if i < len(pos) else ("PUNCT" if _is_punct_like_token(tok) else "X")
        base_lemma = lemmas[i] if i < len(lemmas) else (str(tok or "").strip() or "_")
        tok, norm_pos, norm_lemma = _normalize_token_fields(tok, base_pos, base_lemma)
        ner_value = ner_view[i] if i < len(ner_view) else "O"
        NLP_TOKENS.append(
            {
                "filename": filename,
                "page_index": page_index,
                "sent_index": sent_index,
                "tok_index": i,
                "token": tok,
                "pos": norm_pos,
                "lemma": norm_lemma,
                "ner": ner_value,
                "lang": lang,
            }
        )

NLP_LANGUAGE = ",".join(sorted(_lang_counter.keys())) if _lang_counter else None
NLP_LANGUAGE_STATS = dict(_lang_counter)
DETECTED_LANGUAGES = sorted(_lang_counter.keys())

# Compat descendante: anciennes cles conservees, mais alimentees par la sortie combinee.
NLP_POS = NLP_TOKENS
NLP_LEMMA = NLP_TOKENS
