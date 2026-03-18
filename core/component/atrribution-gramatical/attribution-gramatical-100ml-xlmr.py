from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", flags=re.UNICODE)
TOKEN_RE = re.compile(
    r"[A-Za-z0-9_\u00C0-\u024F]+|[\u0600-\u06FF]+|[^\w\s]",
    re.UNICODE,
)


FR_HINT = {
    "le", "la", "les", "des", "une", "un", "est", "avec", "pour", "dans", "sur",
    "facture", "date", "total", "tva", "montant", "contrat", "article",
}
EN_HINT = {
    "the", "and", "to", "of", "in", "is", "for", "with", "invoice", "date",
    "total", "vat", "amount", "contract", "agreement",
}

PRONOUNS = {
    "fr": {"je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on", "me", "te", "se", "lui", "leur"},
    "en": {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"},
    "ar": {"انا", "أنت", "انت", "هو", "هي", "نحن", "هم", "هن", "انا", "إياك", "اياك"},
}
DETERMINERS = {
    "fr": {"le", "la", "les", "un", "une", "des", "ce", "cet", "cette", "ces"},
    "en": {"the", "a", "an", "this", "that", "these", "those"},
    "ar": {"ال", "هذا", "هذه", "ذلك", "تلك", "هؤلاء"},
}
ADPOSITIONS = {
    "fr": {"de", "du", "des", "a", "au", "aux", "dans", "sur", "sous", "avec", "sans", "pour", "par", "chez"},
    "en": {"of", "in", "on", "at", "by", "for", "from", "with", "without", "to", "into", "over", "under"},
    "ar": {"من", "إلى", "الى", "في", "على", "عن", "مع", "ب", "ل"},
}
CONJUNCTIONS = {
    "fr": {"et", "ou", "mais", "donc", "or", "ni", "car"},
    "en": {"and", "or", "but", "so", "yet", "nor"},
    "ar": {"و", "او", "أو", "لكن", "بل", "ثم"},
}
AUXILIARIES = {
    "fr": {"etre", "est", "sont", "etait", "étaient", "a", "ont", "avais", "avez"},
    "en": {"be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did"},
    "ar": {"كان", "كانت", "يكون", "تكون", "ليس", "ليست"},
}
PARTICLES = {
    "fr": {"ne", "pas", "n", "ni"},
    "en": {"not", "n't", "to"},
    "ar": {"لا", "لم", "لن", "ما"},
}

POS_PROTOTYPE_SEEDS: Dict[str, List[str]] = {
    "NOUN": [
        "contrat", "facture", "document", "montant", "date", "invoice", "agreement", "amount", "car", "engine",
        "سيارة", "عقد", "فاتورة",
    ],
    "VERB": [
        "etre", "avoir", "faire", "do", "make", "is", "are", "run", "sign", "payer", "pay", "go",
        "كان", "تكون", "يوقع",
    ],
    "ADJ": [
        "important", "principal", "juridique", "legal", "technical", "fast", "rapide", "valide",
        "مهم", "قانوني",
    ],
    "ADV": [
        "tres", "plus", "moins", "rapidement", "souvent", "very", "more", "less", "quickly", "often",
        "جدا", "غالبا",
    ],
    "PROPN": [
        "audi", "volkswagen", "lamborghini", "mourad", "jean", "paris", "alger", "europe",
        "أودي", "باريس",
    ],
}


def _env_true(name: str) -> bool:
    raw = str(os.environ.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_offline_mode() -> bool:
    if _env_true("HF_HUB_OFFLINE"):
        return True
    if _env_true("TRANSFORMERS_OFFLINE"):
        return True
    if _env_true("LANG_PIPE_OFFLINE"):
        return True
    return False


def _resolve_model_for_loading(model_name: str) -> tuple[str, bool, str]:
    """
    Returns (model_ref, local_files_only, note).
    - model_ref: local directory or remote model id
    - local_files_only: pass to transformers.from_pretrained
    - note: audit note for terminal/context
    """
    raw_name = str(model_name or "").strip() or "xlm-roberta-base"

    explicit_local = str(globals().get("ML100_MODEL_LOCAL_DIR") or os.environ.get("ML100_MODEL_LOCAL_DIR") or "").strip()
    if explicit_local:
        local_path = Path(explicit_local).expanduser().resolve()
        if local_path.exists():
            return str(local_path), True, f"local-dir:{local_path}"

    as_path = Path(raw_name).expanduser()
    if as_path.exists():
        return str(as_path.resolve()), True, f"local-path:{as_path.resolve()}"

    if _is_offline_mode():
        return raw_name, True, "offline-mode:local-only"

    cache_dir_raw = str(
        globals().get("ML100_MODEL_CACHE_DIR")
        or os.environ.get("ML100_MODEL_CACHE_DIR")
        or (Path(__file__).resolve().parent / ".hf_model_cache")
    ).strip()
    cache_dir = Path(cache_dir_raw).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        local_snap = snapshot_download(
            repo_id=raw_name,
            cache_dir=str(cache_dir),
            local_files_only=False,
            resume_download=True,
        )
        return str(Path(local_snap).resolve()), True, f"auto-installed:{raw_name}"
    except Exception:
        # Fallback: let transformers attempt download directly from model id.
        return raw_name, False, f"remote-hub:{raw_name}"


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


def _strip_accents(text: str) -> str:
    val = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in val if not unicodedata.combining(ch))


def _norm_token(text: str) -> str:
    val = str(text or "").strip().lower()
    val = val.replace("’", "'").replace("`", "'")
    return val


def detect_lang(text: str) -> str:
    t = text or ""
    if AR_RE.search(t):
        return "ar"
    words = [w.lower() for w in WORD_RE.findall(t[:8000])]
    if not words:
        return "en"
    fr_score = sum(1 for w in words if w in FR_HINT)
    en_score = sum(1 for w in words if w in EN_HINT)
    if re.search(r"[éèêàùçôîï]", t.lower()):
        fr_score += 1
    return "fr" if fr_score >= en_score else "en"


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
    Yield dict:
    {
      filename, page_index, sent_index, text, lang
    }
    """
    if data is None:
        return

    if isinstance(data, list) and data and isinstance(data[0], dict) and "pages" in data[0]:
        for d_i, doc in enumerate(data):
            filename = doc.get("filename") or doc.get("doc_id") or f"doc#{d_i}"
            pages = doc.get("pages") or []
            for p_i, page in enumerate(pages):
                page_index = page.get("page_index", page.get("page", p_i + 1))
                sent_items = page.get("sentences_layout") or page.get("sentences") or page.get("chunks") or []
                for s_i, sent in enumerate(sent_items):
                    if isinstance(sent, dict):
                        if sent.get("is_sentence") is False:
                            continue
                        text = str(sent.get("text") or "")
                        lang = str(sent.get("lang") or page.get("lang") or "")
                    else:
                        text = str(sent)
                        lang = str(page.get("lang") or "")
                    text = text.strip()
                    if not text:
                        continue
                    yield {
                        "filename": filename,
                        "page_index": page_index,
                        "sent_index": s_i,
                        "text": text,
                        "lang": lang or detect_lang(text),
                    }
        return

    if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
        for i, d in enumerate(data):
            text = str(d.get("text") or "").strip()
            if not text:
                continue
            filename = d.get("filename") or d.get("doc_id") or f"doc#{i}"
            yield {
                "filename": filename,
                "page_index": None,
                "sent_index": 0,
                "text": text,
                "lang": detect_lang(text),
            }
        return

    if isinstance(data, dict) and "text" in data:
        text = str(data.get("text") or "").strip()
        if text:
            filename = data.get("filename") or data.get("doc_id") or "doc"
            yield {
                "filename": filename,
                "page_index": None,
                "sent_index": 0,
                "text": text,
                "lang": detect_lang(text),
            }
        return

    if isinstance(data, str):
        text = data.strip()
        if text:
            yield {
                "filename": "text",
                "page_index": None,
                "sent_index": 0,
                "text": text,
                "lang": detect_lang(text),
            }
        return

    raise TypeError(f"Format d'entree non supporte: {type(data)}")


def _basic_tokenize(text: str) -> List[str]:
    return [tok for tok in TOKEN_RE.findall(str(text or "")) if str(tok).strip()]


def _normalize_lemma(token: str, lang: str) -> str:
    tok = str(token or "").strip()
    if not tok:
        return "_"
    if _is_punct_like_token(tok):
        return tok
    if lang == "ar":
        return tok
    return _strip_accents(tok).lower()


def _guess_pos(token: str, lang: str, prev_token: str = "", next_token: str = "") -> str:
    tok = str(token or "").strip()
    if not tok:
        return "X"
    if _is_punct_like_token(tok):
        return "PUNCT"
    if re.fullmatch(r"[+\-]?\d+(?:[.,]\d+)?", tok):
        return "NUM"

    low = _norm_token(tok)
    low_no_acc = _norm_token(_strip_accents(tok))
    prev_low = _norm_token(prev_token)

    if low in PRONOUNS.get(lang, set()) or low_no_acc in PRONOUNS.get(lang, set()):
        return "PRON"
    if low in DETERMINERS.get(lang, set()) or low_no_acc in DETERMINERS.get(lang, set()):
        return "DET"
    if low in ADPOSITIONS.get(lang, set()) or low_no_acc in ADPOSITIONS.get(lang, set()):
        return "ADP"
    if low in CONJUNCTIONS.get(lang, set()) or low_no_acc in CONJUNCTIONS.get(lang, set()):
        return "CCONJ"
    if low in AUXILIARIES.get(lang, set()) or low_no_acc in AUXILIARIES.get(lang, set()):
        return "AUX"
    if low in PARTICLES.get(lang, set()) or low_no_acc in PARTICLES.get(lang, set()):
        return "PART"

    if lang == "fr" and re.search(r"(er|ir|re|ant|ait|ent|ons|ez)$", low_no_acc):
        return "VERB"
    if lang == "en" and re.search(r"(ing|ed|ize|ise|fy|en|s)$", low_no_acc):
        return "VERB"
    if lang == "ar" and re.search(r"^(ي|ت|ن|ا).+", low):
        return "VERB"

    if tok[0].isupper() and prev_low not in {"", ".", "!", "?"}:
        return "PROPN"

    if re.search(r"(ive|ous|able|ible|al|el|ique|ary|ory|less|ful)$", low_no_acc):
        return "ADJ"
    if (lang == "fr" and low_no_acc.endswith("ment")) or (lang == "en" and low_no_acc.endswith("ly")):
        return "ADV"

    return "NOUN"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return 0.0
    if a.size == 0 or b.size == 0:
        return 0.0
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0:
        return 0.0
    return num / den


def _build_pos_prototypes(encoder: "_XLMRContextEncoder") -> Dict[str, np.ndarray]:
    items: List[tuple[str, str]] = []
    for tag, seeds in POS_PROTOTYPE_SEEDS.items():
        for seed in seeds:
            if str(seed or "").strip():
                items.append((tag, seed.strip()))
    if not items:
        return {}

    token_lists = [[token] for _, token in items]
    vectors = encoder.encode_token_lists(token_lists)
    grouped: Dict[str, List[np.ndarray]] = {}
    for idx, (tag, _) in enumerate(items):
        if idx >= len(vectors):
            continue
        mat = vectors[idx]
        if not isinstance(mat, np.ndarray) or mat.size == 0:
            continue
        vec = np.asarray(mat[0], dtype=np.float32)
        if vec.size == 0:
            continue
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        grouped.setdefault(tag, []).append(vec)

    out: Dict[str, np.ndarray] = {}
    for tag, rows in grouped.items():
        if not rows:
            continue
        avg = np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(avg))
        if norm > 0:
            avg = avg / norm
        out[tag] = avg
    return out


def _refine_pos_tags(
    tokens: List[str],
    lemmas: List[str],
    pos_tags: List[str],
    lang: str,
    token_vectors: Optional[np.ndarray],
    pos_prototypes: Dict[str, np.ndarray],
    use_vector_refine: bool,
) -> tuple[List[str], Dict[str, int]]:
    refined = list(pos_tags or [])
    if len(refined) < len(tokens):
        refined.extend(["NOUN"] * (len(tokens) - len(refined)))

    changed = 0
    for idx, tok in enumerate(tokens):
        tok_txt = str(tok or "").strip()
        if not tok_txt:
            continue
        if _is_punct_like_token(tok_txt):
            if refined[idx] != "PUNCT":
                refined[idx] = "PUNCT"
                changed += 1
            continue
        if re.fullmatch(r"[+\-]?\d+(?:[.,]\d+)?", tok_txt):
            if refined[idx] != "NUM":
                refined[idx] = "NUM"
                changed += 1
            continue

        low = _norm_token(tok_txt)
        low_no_acc = _norm_token(_strip_accents(tok_txt))
        prev_tok = tokens[idx - 1] if idx > 0 else ""
        prev_pos = refined[idx - 1] if idx > 0 else ""
        next_tok = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        curr = refined[idx]

        rule_tag = ""
        if low in PRONOUNS.get(lang, set()) or low_no_acc in PRONOUNS.get(lang, set()):
            rule_tag = "PRON"
        elif low in DETERMINERS.get(lang, set()) or low_no_acc in DETERMINERS.get(lang, set()):
            rule_tag = "DET"
        elif low in ADPOSITIONS.get(lang, set()) or low_no_acc in ADPOSITIONS.get(lang, set()):
            rule_tag = "ADP"
        elif low in CONJUNCTIONS.get(lang, set()) or low_no_acc in CONJUNCTIONS.get(lang, set()):
            rule_tag = "CCONJ"
        elif low in AUXILIARIES.get(lang, set()) or low_no_acc in AUXILIARIES.get(lang, set()):
            rule_tag = "AUX"
        elif low in PARTICLES.get(lang, set()) or low_no_acc in PARTICLES.get(lang, set()):
            rule_tag = "PART"
        elif (lang == "fr" and low_no_acc.endswith("ment")) or (lang == "en" and low_no_acc.endswith("ly")):
            rule_tag = "ADV"
        elif curr in {"NOUN", "ADJ", "X"} and prev_pos == "DET":
            rule_tag = "NOUN"
        elif curr in {"NOUN", "ADJ", "X"} and prev_pos in {"AUX", "PRON", "PART"}:
            guess = _guess_pos(tok_txt, lang, prev_tok, next_tok)
            if guess == "VERB":
                rule_tag = "VERB"

        cand = rule_tag or curr

        if idx == 0 and cand == "PROPN" and tok_txt[:1].isupper():
            if low_no_acc in DETERMINERS.get(lang, set()) or low_no_acc in PRONOUNS.get(lang, set()):
                cand = "DET" if low_no_acc in DETERMINERS.get(lang, set()) else "PRON"

        if use_vector_refine and isinstance(token_vectors, np.ndarray) and idx < token_vectors.shape[0] and pos_prototypes:
            vec = np.asarray(token_vectors[idx], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
                best_tag = ""
                best_score = -1.0
                cand_score = _cosine(vec, pos_prototypes.get(cand, np.zeros(0, dtype=np.float32))) if cand else -1.0
                for tag, proto in pos_prototypes.items():
                    score = _cosine(vec, proto)
                    if score > best_score:
                        best_score = score
                        best_tag = tag
                # Seuil + marge pour eviter des bascules instables.
                if best_tag and best_score >= 0.20 and (best_score - cand_score) >= 0.08:
                    if not (cand == "PROPN" and best_tag == "NOUN" and tok_txt[:1].isupper() and idx > 0):
                        cand = best_tag

        if cand != curr:
            refined[idx] = cand
            changed += 1

    return refined, {"changed": changed, "total": len(tokens)}


def _hash_vec(text: str, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    token = str(text or "").strip().lower() or "_"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    idx = int.from_bytes(digest[:8], "big") % dim
    sign = 1.0 if (digest[8] & 1) == 0 else -1.0
    out[idx] = sign
    return out


class _XLMRContextEncoder:
    def __init__(self, model_name: str, max_length: int, batch_size: int, fallback_dim: int):
        self.model_name_requested = str(model_name or "xlm-roberta-base")
        self.model_name = self.model_name_requested
        self.max_length = max(16, int(max_length))
        self.batch_size = max(1, int(batch_size))
        self.dim = max(64, int(fallback_dim))
        self.backend = "hash-fallback"
        self.error: Optional[str] = None
        self.install_note = ""
        self.local_files_only = False

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            resolved_model, local_only, note = _resolve_model_for_loading(self.model_name_requested)
            self.model_name = resolved_model
            self.local_files_only = bool(local_only)
            self.install_note = note

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            hidden_size = getattr(getattr(self._model, "config", None), "hidden_size", None)
            if hidden_size:
                self.dim = int(hidden_size)
            self.backend = f"xlmr:{self.model_name_requested}"
        except Exception as exc:
            self.error = str(exc)
            self._torch = None
            self._tokenizer = None
            self._model = None

    def encode_token_lists(self, token_lists: List[List[str]]) -> List[np.ndarray]:
        if not token_lists:
            return []

        if self._torch is None or self._tokenizer is None or self._model is None:
            return [np.stack([_hash_vec(tok, self.dim) for tok in toks], axis=0) if toks else np.zeros((0, self.dim), dtype=np.float32) for toks in token_lists]

        torch = self._torch
        out: List[np.ndarray] = []
        try:
            with torch.inference_mode():
                for i in range(0, len(token_lists), self.batch_size):
                    batch = token_lists[i:i + self.batch_size]
                    enc = self._tokenizer(
                        batch,
                        is_split_into_words=True,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    model_inputs = {k: v.to(self._device) for k, v in enc.items()}
                    outputs = self._model(**model_inputs)
                    hidden = getattr(outputs, "last_hidden_state", None)
                    if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
                        hidden = outputs[0]
                    if hidden is None:
                        for toks in batch:
                            rows = np.stack([_hash_vec(tok, self.dim) for tok in toks], axis=0) if toks else np.zeros((0, self.dim), dtype=np.float32)
                            out.append(rows)
                        continue

                    hidden_np = hidden.detach().cpu().numpy().astype(np.float32)
                    for b_idx, toks in enumerate(batch):
                        word_ids = enc.word_ids(batch_index=b_idx)
                        groups: Dict[int, List[np.ndarray]] = {}
                        for tok_idx, wid in enumerate(word_ids):
                            if wid is None or wid < 0 or wid >= len(toks):
                                continue
                            groups.setdefault(wid, []).append(hidden_np[b_idx, tok_idx])

                        rows: List[np.ndarray] = []
                        for wid in range(len(toks)):
                            vectors = groups.get(wid)
                            if not vectors:
                                rows.append(_hash_vec(toks[wid], self.dim))
                                continue
                            vec = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
                            norm = float(np.linalg.norm(vec))
                            if norm > 0:
                                vec = vec / norm
                            rows.append(vec)
                        out.append(np.stack(rows, axis=0) if rows else np.zeros((0, self.dim), dtype=np.float32))
        except Exception as exc:
            self.error = str(exc)
            self.backend = "hash-fallback"
            return [np.stack([_hash_vec(tok, self.dim) for tok in toks], axis=0) if toks else np.zeros((0, self.dim), dtype=np.float32) for toks in token_lists]

        if len(out) < len(token_lists):
            for toks in token_lists[len(out):]:
                rows = np.stack([_hash_vec(tok, self.dim) for tok in toks], axis=0) if toks else np.zeros((0, self.dim), dtype=np.float32)
                out.append(rows)
        return out


def _heuristic_ner(token: str, lemma: str, pos: str, lang: str) -> str:
    tok = str(token or "")
    lem = str(lemma or "")
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", tok):
        return "B-DATE"
    if "@" in tok and "." in tok:
        return "B-EMAIL"
    if re.fullmatch(r"\+?\d{7,15}", re.sub(r"[^\d+]", "", tok)):
        return "B-PHONE"
    if pos == "PROPN":
        if lang == "ar":
            return "B-ORG"
        return "B-PER"
    if lem.upper() in {"EUR", "USD", "DZD", "TTC", "TVA"}:
        return "B-MISC"
    return "O"


def _run() -> None:
    data = get_previous_cell_input()
    if data is None:
        raise RuntimeError("Je ne trouve pas de donnees d'entree pour le composant grammaire 100ml.")

    model_name = str(globals().get("ML100_MODEL_NAME") or os.environ.get("ML100_MODEL_NAME") or "xlm-roberta-base")
    max_length = int(globals().get("ML100_MAX_LENGTH") or os.environ.get("ML100_MAX_LENGTH") or 256)
    batch_size = int(globals().get("ML100_BATCH_SIZE") or os.environ.get("ML100_BATCH_SIZE") or 8)
    fallback_dim = int(globals().get("ML100_HASH_FALLBACK_DIM") or os.environ.get("ML100_HASH_FALLBACK_DIM") or 384)

    rows = list(iter_sentences_from_input(data))
    token_lists: List[List[str]] = [_basic_tokenize(row["text"]) for row in rows]

    encoder = _XLMRContextEncoder(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        fallback_dim=fallback_dim,
    )
    embeddings = encoder.encode_token_lists(token_lists)

    nlp_analyses: List[Dict[str, Any]] = []
    nlp_sentences: List[Dict[str, Any]] = []
    nlp_entities: List[Dict[str, Any]] = []
    nlp_tokens: List[Dict[str, Any]] = []
    lang_counter: Counter = Counter()
    pos_counter: Counter = Counter()
    pos_refined_count = 0
    pos_total_count = 0
    use_vector_refine = encoder.backend.startswith("xlmr:")
    pos_prototypes = _build_pos_prototypes(encoder) if use_vector_refine else {}

    for idx, row in enumerate(rows):
        filename = row["filename"]
        page_index = row["page_index"]
        sent_index = row["sent_index"]
        text = row["text"]
        lang = str(row.get("lang") or detect_lang(text))
        tokens = token_lists[idx] if idx < len(token_lists) else _basic_tokenize(text)
        vecs = embeddings[idx] if idx < len(embeddings) else np.zeros((len(tokens), encoder.dim), dtype=np.float32)

        norm_tokens: List[str] = []
        lemmas: List[str] = []
        base_pos_tags: List[str] = []
        ner_labels: List[str] = []
        sentence_entities_flat: List[Dict[str, Any]] = []

        for tok_i, tok in enumerate(tokens):
            prev_tok = tokens[tok_i - 1] if tok_i > 0 else ""
            next_tok = tokens[tok_i + 1] if tok_i + 1 < len(tokens) else ""
            pos_val = _guess_pos(tok, lang, prev_tok, next_tok)
            lemma_val = _normalize_lemma(tok, lang)
            tok_norm, pos_val, lemma_val = _normalize_token_fields(tok, pos_val, lemma_val)
            norm_tokens.append(tok_norm)
            base_pos_tags.append(pos_val)
            lemmas.append(lemma_val)

        pos_tags, refine_stats = _refine_pos_tags(
            tokens=norm_tokens,
            lemmas=lemmas,
            pos_tags=base_pos_tags,
            lang=lang,
            token_vectors=vecs if isinstance(vecs, np.ndarray) else None,
            pos_prototypes=pos_prototypes,
            use_vector_refine=use_vector_refine,
        )
        pos_refined_count += int(refine_stats.get("changed") or 0)
        pos_total_count += int(refine_stats.get("total") or 0)

        for tok_i, tok_norm in enumerate(norm_tokens):
            pos_val = pos_tags[tok_i] if tok_i < len(pos_tags) else "NOUN"
            lemma_val = lemmas[tok_i] if tok_i < len(lemmas) else _normalize_lemma(tok_norm, lang)
            ner_val = _heuristic_ner(tok_norm, lemma_val, pos_val, lang)
            ner_labels.append(ner_val)
            pos_counter[pos_val] += 1

            nlp_tokens.append(
                {
                    "filename": filename,
                    "page_index": page_index,
                    "sent_index": sent_index,
                    "tok_index": tok_i,
                    "token": tok_norm,
                    "pos": pos_val,
                    "lemma": lemma_val,
                    "ner": ner_val,
                    "lang": lang,
                    "xlmr_backend": encoder.backend,
                }
            )

            if ner_val != "O":
                ent_row = {
                    "type": ner_val.replace("B-", "").replace("I-", ""),
                    "text": tok_norm,
                }
                sentence_entities_flat.append(ent_row)
                nlp_entities.append(
                    {
                        "filename": filename,
                        "page_index": page_index,
                        "sent_index": sent_index,
                        "lang": lang,
                        "type": ent_row["type"],
                        "text": ent_row["text"],
                    }
                )

        nlp_analyses.append(
            {
                "filename": filename,
                "doc": filename,
                "page_index": page_index,
                "sent_index": sent_index,
                "lang": lang,
                "text": text,
                "tokens": norm_tokens,
                "lemmas": lemmas,
                "pos": pos_tags,
                "ner_labels": ner_labels,
                "entities": {"flat": sentence_entities_flat},
                "xlmr_backend": encoder.backend,
                "xlmr_vector_dim": int(encoder.dim),
                "xlmr_token_vectors_count": int(vecs.shape[0]) if isinstance(vecs, np.ndarray) else len(tokens),
            }
        )
        nlp_sentences.append(
            {
                "filename": filename,
                "page_index": page_index,
                "sent_index": sent_index,
                "lang": lang,
                "text": text,
            }
        )
        lang_counter[lang] += 1

    NLP_ANALYSES = nlp_analyses
    NLP_SENTENCES = nlp_sentences
    NLP_ENTITIES = nlp_entities
    NLP_TOKENS = nlp_tokens
    NLP_POS = NLP_TOKENS
    NLP_LEMMA = NLP_TOKENS
    NLP_LANGUAGE = ",".join(sorted(lang_counter.keys())) if lang_counter else None
    NLP_LANGUAGE_STATS = dict(lang_counter)
    DETECTED_LANGUAGES = sorted(lang_counter.keys())
    NLP_GRAMMAR_BACKEND = encoder.backend
    NLP_GRAMMAR_MODEL = encoder.model_name_requested
    NLP_GRAMMAR_MODEL_SOURCE = encoder.model_name
    NLP_GRAMMAR_MODEL_INSTALL = encoder.install_note
    NLP_GRAMMAR_ERROR = encoder.error
    NLP_POS_METHOD = "hybrid-rules+context+xlmr-prototypes-v2" if use_vector_refine else "hybrid-rules+context-v2"
    NLP_POS_REFINED_COUNT = int(pos_refined_count)
    NLP_POS_TOTAL = int(pos_total_count)
    NLP_POS_REFINED_RATE = (float(pos_refined_count) / float(pos_total_count)) if pos_total_count > 0 else 0.0
    NLP_POS_DISTRIBUTION = dict(pos_counter)
    NLP_POS_TOP = [{"pos": tag, "count": count} for tag, count in pos_counter.most_common(10)]

    globals().update(
        {
            "NLP_ANALYSES": NLP_ANALYSES,
            "NLP_SENTENCES": NLP_SENTENCES,
            "NLP_ENTITIES": NLP_ENTITIES,
            "NLP_TOKENS": NLP_TOKENS,
            "NLP_POS": NLP_POS,
            "NLP_LEMMA": NLP_LEMMA,
            "NLP_LANGUAGE": NLP_LANGUAGE,
            "NLP_LANGUAGE_STATS": NLP_LANGUAGE_STATS,
            "DETECTED_LANGUAGES": DETECTED_LANGUAGES,
            "NLP_GRAMMAR_BACKEND": NLP_GRAMMAR_BACKEND,
            "NLP_GRAMMAR_MODEL": NLP_GRAMMAR_MODEL,
            "NLP_GRAMMAR_MODEL_SOURCE": NLP_GRAMMAR_MODEL_SOURCE,
            "NLP_GRAMMAR_MODEL_INSTALL": NLP_GRAMMAR_MODEL_INSTALL,
            "NLP_GRAMMAR_ERROR": NLP_GRAMMAR_ERROR,
            "NLP_POS_METHOD": NLP_POS_METHOD,
            "NLP_POS_REFINED_COUNT": NLP_POS_REFINED_COUNT,
            "NLP_POS_TOTAL": NLP_POS_TOTAL,
            "NLP_POS_REFINED_RATE": NLP_POS_REFINED_RATE,
            "NLP_POS_DISTRIBUTION": NLP_POS_DISTRIBUTION,
            "NLP_POS_TOP": NLP_POS_TOP,
        }
    )

    print(
        "[grammar-100ml-xlmr] "
        f"sentences={len(NLP_SENTENCES)} | tokens={len(NLP_TOKENS)} | "
        f"langs={DETECTED_LANGUAGES} | backend={NLP_GRAMMAR_BACKEND} | "
        f"model={NLP_GRAMMAR_MODEL} | source={NLP_GRAMMAR_MODEL_SOURCE}"
    )
    if NLP_GRAMMAR_MODEL_INSTALL:
        print(f"[grammar-100ml-xlmr][model] {NLP_GRAMMAR_MODEL_INSTALL}")
    if NLP_GRAMMAR_ERROR:
        print(f"[grammar-100ml-xlmr][warn] fallback actif: {NLP_GRAMMAR_ERROR}")
    top_tags = ", ".join(f"{row['pos']}:{row['count']}" for row in NLP_POS_TOP) or "n/a"
    print(
        "[grammar-100ml-xlmr][pos] "
        f"method={NLP_POS_METHOD} | refined={NLP_POS_REFINED_COUNT}/{NLP_POS_TOTAL} | "
        f"top={top_tags}"
    )


_run()
