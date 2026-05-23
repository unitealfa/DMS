from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+", re.UNICODE)

FR_STOP = {
    "a", "à", "afin", "ainsi", "alors", "au", "aucun", "aux", "avec", "avoir", "bon",
    "car", "ce", "cela", "ces", "cet", "cette", "chez", "comme", "comment", "dans",
    "de", "des", "deux", "du", "elle", "elles", "en", "est", "et", "être", "fait",
    "facture", "il", "ils", "je", "la", "le", "les", "leur", "leurs", "lors", "mais",
    "montant", "nous", "ou", "où", "par", "pas", "plus", "pour", "que", "qui", "sans",
    "sera", "ses", "si", "son", "sont", "sur", "tous", "tout", "très", "tva", "un",
    "une", "vous", "votre",
}

EN_STOP = {
    "a", "about", "after", "all", "also", "amount", "an", "and", "any", "are", "as",
    "at", "be", "been", "before", "between", "by", "can", "contract", "date", "do",
    "document", "due", "for", "from", "has", "have", "he", "her", "his", "in",
    "invoice", "is", "it", "its", "must", "of", "on", "or", "our", "shall", "she",
    "should", "that", "the", "their", "these", "this", "to", "total", "vat", "was",
    "we", "were", "will", "with", "you", "your",
}

FR_ACCENT_RE = re.compile(r"[àâçéèêëîïôùûüÿœæ]", re.I)
EN_MARKER_RE = re.compile(r"\b(the|and|invoice|agreement|amount|due|shall|must|will|with|from|this|that)\b", re.I)
FR_MARKER_RE = re.compile(r"\b(le|la|les|des|une|avec|pour|dans|facture|montant|contrat|article|sera|sont|être|tva)\b", re.I)


def _strip_accents(value: str) -> str:
    norm = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


FR_STOP_NORM = {_strip_accents(w.lower()) for w in FR_STOP}
EN_STOP_NORM = {_strip_accents(w.lower()) for w in EN_STOP}


def normalize_lang_code(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw or raw in {"none", "null", "nan", "n/a", "und", "unknown"}:
        return "unknown"
    raw = raw.replace("_", "-")
    if raw.startswith(("fr", "fra", "fre")):
        return "fr"
    if raw.startswith(("en", "eng")):
        return "en"
    if raw.startswith(("ar", "ara")):
        return "ar"
    return raw.split("-", 1)[0] if raw else "unknown"


def language_scores(text: str) -> Dict[str, float]:
    value = str(text or "")
    compact = value.strip()
    if not compact:
        return {"unknown": 1.0}

    ar_chars = len(AR_RE.findall(compact))
    latin_chars = sum(1 for ch in compact if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("\u00C0" <= ch <= "\u024F"))
    alpha_total = max(1, ar_chars + latin_chars)

    scores: Counter = Counter()
    if ar_chars:
        scores["ar"] += ar_chars * 2.2
        if ar_chars / alpha_total >= 0.18 or (ar_chars >= 4 and latin_chars == 0):
            scores["ar"] += 18

    words = []
    for token in TOKEN_RE.findall(compact[:12000]):
        if AR_RE.search(token):
            scores["ar"] += 2
            continue
        norm = _strip_accents(token.lower())
        if len(norm) <= 1:
            continue
        words.append(norm)

    for word in words:
        if word in FR_STOP_NORM:
            scores["fr"] += 3
        if word in EN_STOP_NORM:
            scores["en"] += 3

    # Indices forts mais pas absolus: utiles sur textes courts/OCR bruité.
    scores["fr"] += len(FR_ACCENT_RE.findall(compact)) * 3.5
    scores["fr"] += len(FR_MARKER_RE.findall(compact)) * 4
    scores["en"] += len(EN_MARKER_RE.findall(compact)) * 4

    # Terminaisons fréquentes. Faible poids pour éviter les faux positifs.
    for word in words:
        if word.endswith(("tion", "ment", "eur", "euse", "ique")):
            scores["fr"] += 0.45
        if word.endswith(("ing", "ed", "ness", "ity", "tion")):
            scores["en"] += 0.45

    if not scores:
        return {"unknown": 1.0}
    return {lang: float(score) for lang, score in scores.items() if score > 0}


def detect_lang(text: str, *, default: str = "unknown") -> str:
    scores = language_scores(text)
    if not scores or "unknown" in scores:
        return default
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_lang, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score < 2.5:
        return default
    if top_lang in {"fr", "en"} and second_score and (top_score - second_score) < 2.0:
        # Sur égalité FR/EN, les accents tranchent. Sinon on évite de sur-déclarer FR.
        return "fr" if FR_ACCENT_RE.search(str(text or "")) else "en"
    return top_lang


def detect_languages_from_chunks(chunks: Iterable[str], *, min_score: float = 3.0) -> Tuple[List[str], Dict[str, int]]:
    counter: Counter = Counter()
    for chunk in chunks:
        scores = language_scores(str(chunk or ""))
        for lang, score in scores.items():
            lang = normalize_lang_code(lang)
            if lang == "unknown" or score < min_score:
                continue
            counter[lang] += max(1, int(round(score)))
    langs = [lang for lang, _ in counter.most_common()]
    return langs, {lang: int(counter[lang]) for lang in langs}
