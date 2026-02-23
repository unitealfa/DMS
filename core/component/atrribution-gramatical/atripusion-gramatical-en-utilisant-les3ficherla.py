# =========================
# 1) Chemin vers tes .py
# =========================
import sys, types, re, importlib, os
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

def detect_lang(text: str) -> str:
    t = text or ""
    if _AR_RE.search(t):
        return "ar"
    words = [w.lower() for w in _WORD_RE.findall(t[:8000])]
    if not words:
        return "en"
    fr_score = sum(1 for w in words if w in _FR_HINT)
    en_score = sum(1 for w in words if w in _EN_HINT)
    if re.search(r"[éèêàùçôîï]", t.lower()):
        fr_score += 1
    return "fr" if fr_score >= en_score else "en"

def get_previous_cell_input():
    g = globals()
    for k in ("selected", "TOK_DOCS", "FINAL_DOCS", "DOCS", "TEXT_DOCS", "_"):
        if k in g and g[k] is not None:
            return g[k]
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

print("\n" + "="*120)
print("RUN EN (engcode.py)")
print("="*120)
engcode.run_from_previous_cell(data=data, max_sentences=MAX_SENTENCES_PER_LANG)

print("\n" + "="*120)
print("RUN FR (frcode.py)")
print("="*120)
frcode.run_from_previous_cell(data=data, max_sentences=MAX_SENTENCES_PER_LANG)

if HAVE_AR:
    print("\n" + "="*120)
    print("RUN AR (arabcode.py)")
    print("="*120)
    arabcode.run_from_previous_cell(data=data, max_sentences=MAX_SENTENCES_PER_LANG)
