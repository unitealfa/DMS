from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u00C0-\u024F\u0600-\u06FF]+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\:\;\n])\s+")

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "will", "shall", "must",
    "a", "an", "to", "of", "in", "on", "at", "as", "by", "is", "be", "been", "being",
    "dans", "avec", "pour", "par", "les", "des", "une", "sur", "aux", "est", "sont", "sera", "etre", "ce",
    "de", "du", "la", "le", "un", "en", "et", "ou", "au", "mais", "donc", "or", "ni", "car",
    "qui", "que", "quoi", "dont", "où", "ou", "plus", "moins", "très", "tres",
    "tout", "tous", "toute", "toutes", "son", "sa", "ses", "leur", "leurs",
    "notre", "nos", "votre", "vos", "ainsi", "aussi", "alors", "donc", "or", "ni", "car",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "moi", "toi", "se",
    "i", "you", "he", "she", "it", "we", "they", "who", "which", "what", "where", "when",
    "في", "من", "على", "الى", "إلى", "عن", "مع", "و", "او", "أو", "هذا", "هذه",
    "page", "pages", "document", "documents", "corpus", "test", "tests", "file", "files",
}

NOISE_TERMS = {
    "chose", "truc", "element", "elements", "partie", "parties", "item", "items", "etc",
    "hereby", "thereof", "whereas", "herein", "hereinafter", "therein",
    "every", "english", "period", "matter", "subject",
}

ALLOWED_SHORT_TERMS = {"tva", "ttc", "ht", "loi", "dzd", "eur", "usd"}

SEMANTIC_POS_ALLOWED = {
    "NOUN", "PROPN", "ADJ",
    "NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "CD",
}
SEMANTIC_POS_ALLOWED_PREFIX = ("NOUN", "PROPN", "ADJ", "NN", "JJ")
SEMANTIC_POS_BLOCKED = {
    "PRON", "DET", "ADP", "CCONJ", "SCONJ", "CONJ", "PART", "AUX", "INTJ", "PUNCT", "SYM",
    "ADV", "RB", "RBR", "RBS",
    "PRP", "PRP$", "WP", "WP$", "WDT", "DT", "IN", "TO", "CC", "MD", "UH",
}
SEMANTIC_POS_BLOCKED_PREFIX = ("PRON", "DET", "ADV", "CONJ", "AUX", "ADP", "PUNCT")

TOPIC_SOURCES = ("ML50_TOPICS", "ML100_TOPICS")
MAX_SENTENCES_PER_DOC = 240
MAX_MATCHES_PER_LINK = 14
MIN_SENTENCE_MATCH_SCORE = 0.10
MIN_LINK_SCORE = 0.15
MIN_SHARED_TOPICS = 1
MAX_SHARED_TERMS_PER_MATCH = 8
MIN_INFORMATIVE_TERM_SCORE = 0.85


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_term(value: Any) -> str:
    txt = str(value or "").strip().lower()
    if not txt:
        return ""
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = txt.replace("’", "'").replace("`", "'")
    return txt


def _filename_aliases(filename: Any) -> List[str]:
    raw = str(filename or "").strip()
    if not raw:
        return []
    aliases = [raw, Path(raw).name]
    out: List[str] = []
    seen = set()
    for item in aliases:
        key = _normalize_term(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _sentence_key(filename: str, page_index: int, sent_index: int) -> str:
    return f"{_normalize_term(filename)}|{page_index}|{sent_index}"


def _is_semantic_pos(pos_value: Any, ner_value: Any) -> bool:
    pos = str(pos_value or "").strip().upper()
    if pos in SEMANTIC_POS_BLOCKED:
        return False
    if any(pos.startswith(prefix) for prefix in SEMANTIC_POS_BLOCKED_PREFIX):
        return False
    if pos in SEMANTIC_POS_ALLOWED:
        return True
    if any(pos.startswith(prefix) for prefix in SEMANTIC_POS_ALLOWED_PREFIX):
        return True
    ner = str(ner_value or "").strip().upper()
    if ner and ner not in {"O", "NONE", "NULL"}:
        return True
    return False


def _is_informative_term(term: str) -> bool:
    norm = _normalize_term(term)
    if not norm:
        return False
    if norm in STOPWORDS or norm in NOISE_TERMS:
        return False
    if set(norm) == {"_"}:
        return False
    if not any(ch.isalpha() for ch in norm):
        return False
    if len(norm) < 3:
        return False
    if len(norm) == 3 and norm not in ALLOWED_SHORT_TERMS:
        return False
    return True


def _tokenize_terms(text: str) -> List[str]:
    out: List[str] = []
    for tok in TOKEN_RE.findall(str(text or "")):
        norm = _normalize_term(tok)
        if not _is_informative_term(norm):
            continue
        out.append(norm)
    return out


def _split_to_sentences(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(raw) if str(p or "").strip()]
    if parts:
        return parts
    return [raw]


def _clip_text(text: str, limit: int = 220) -> str:
    txt = str(text or "").strip()
    if len(txt) <= limit:
        return txt
    return txt[: max(0, limit - 3)].rstrip() + "..."


def _doc_key(doc_id: Any, filename: Any, idx: int) -> str:
    sid = str(doc_id or "").strip()
    if sid:
        return f"id:{sid}"
    sfn = str(filename or "").strip()
    if sfn:
        return f"fn:{sfn}"
    return f"idx:{idx}"


def _doc_aliases(doc_id: Any, filename: Any, idx: int) -> List[str]:
    out: Set[str] = set()
    key = _doc_key(doc_id, filename, idx)
    out.add(key)
    sid = str(doc_id or "").strip()
    sfn = str(filename or "").strip()
    if sid:
        out.add(f"id:{sid}")
    if sfn:
        out.add(f"fn:{sfn}")
        out.add(f"fn:{Path(sfn).name}")
    return [x for x in out if x]


def _extract_classification_terms(row: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    doc_type = str(row.get("doc_type") or "").strip()
    for tok in _tokenize_terms(doc_type):
        out.add(tok)

    kw = row.get("keyword_matches")
    if isinstance(kw, dict):
        for key in ("strong", "medium", "weak"):
            for item in _safe_list(kw.get(key)):
                if isinstance(item, dict):
                    raw = item.get("keyword") or item.get("value") or item.get("term")
                else:
                    raw = item
                for tok in _tokenize_terms(str(raw or "")):
                    out.add(tok)
    return out


def _build_classification_index(ctx: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for i, row in enumerate(_safe_list(ctx.get("RESULTS"))):
        if not isinstance(row, dict):
            continue
        terms = _extract_classification_terms(row)
        if not terms:
            continue
        aliases = _doc_aliases(row.get("doc_id"), row.get("filename"), i)
        for alias in aliases:
            out.setdefault(alias, set()).update(terms)
    return out


def _build_semantic_sentence_index(ctx: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    rows = _safe_list(ctx.get("NLP_TOKENS"))
    if not rows:
        return out

    for row in rows:
        if not isinstance(row, dict):
            continue
        if not _is_semantic_pos(row.get("pos"), row.get("ner")):
            continue

        page_index = _safe_int(row.get("page_index"), -1)
        sent_index = _safe_int(row.get("sent_index"), -1)
        if page_index < 0 or sent_index < 0:
            continue

        lemma = _normalize_term(row.get("lemma"))
        token = _normalize_term(row.get("token"))
        term = lemma if _is_informative_term(lemma) else token
        if not _is_informative_term(term):
            continue

        for alias in _filename_aliases(row.get("filename")):
            key = _sentence_key(alias, page_index, sent_index)
            out.setdefault(key, set()).add(term)
    return out


def _lookup_semantic_terms(
    semantic_index: Dict[str, Set[str]],
    filename: str,
    page_index: int,
    sent_index: int,
) -> Set[str]:
    out: Set[str] = set()
    for alias in _filename_aliases(filename):
        values = semantic_index.get(_sentence_key(alias, page_index, sent_index))
        if values:
            out.update(values)
    return out


def _iter_doc_sentences(
    doc: Dict[str, Any],
    filename: str,
    semantic_index: Optional[Dict[str, Set[str]]] = None,
) -> Iterable[Dict[str, Any]]:
    pages = _safe_list(doc.get("pages"))
    for page_pos, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        page_index = _safe_int(page.get("page_index"), page_pos)
        sents = page.get("sentences_layout")
        if isinstance(sents, list) and sents:
            for sent_pos, sent in enumerate(sents):
                text = str(sent.get("text") or "") if isinstance(sent, dict) else str(sent or "")
                text = text.strip()
                if not text:
                    continue
                terms = _tokenize_terms(text)
                if isinstance(semantic_index, dict):
                    semantic_terms = _lookup_semantic_terms(semantic_index, filename, page_index, sent_pos)
                    if semantic_terms:
                        kept = [t for t in terms if t in semantic_terms]
                        terms = kept if kept else sorted(semantic_terms)
                if not terms:
                    continue
                yield {
                    "page_index": page_index,
                    "sent_index": sent_pos,
                    "text": text,
                    "terms": terms,
                    "terms_set": set(terms),
                }
            continue

        page_text = str(page.get("page_text") or page.get("text") or "")
        for sent_pos, sent_text in enumerate(_split_to_sentences(page_text)):
            terms = _tokenize_terms(sent_text)
            if isinstance(semantic_index, dict):
                semantic_terms = _lookup_semantic_terms(semantic_index, filename, page_index, sent_pos)
                if semantic_terms:
                    kept = [t for t in terms if t in semantic_terms]
                    terms = kept if kept else sorted(semantic_terms)
            if not terms:
                continue
            yield {
                "page_index": page_index,
                "sent_index": sent_pos,
                "text": sent_text,
                "terms": terms,
                "terms_set": set(terms),
            }


def _doc_text_score(doc: Dict[str, Any]) -> int:
    total = 0
    filename = str(doc.get("filename") or "")
    for row in _iter_doc_sentences(doc, filename=filename, semantic_index=None):
        total += len(str(row.get("text") or ""))
    return total


def _dedupe_docs(rows: Any) -> List[Dict[str, Any]]:
    docs = [d for d in _safe_list(rows) if isinstance(d, dict)]
    if not docs:
        return []

    best: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for i, doc in enumerate(docs):
        key = _doc_key(doc.get("doc_id"), doc.get("filename"), i)
        if key not in best:
            best[key] = doc
            order.append(key)
            continue
        if _doc_text_score(doc) > _doc_text_score(best[key]):
            best[key] = doc
    return [best[k] for k in order]


def _index_topic_rows(ctx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for source in TOPIC_SOURCES:
        for row in _safe_list(ctx.get(source)):
            if not isinstance(row, dict):
                continue
            aliases = _doc_aliases(row.get("doc_id"), row.get("filename"), 0)
            for alias in aliases:
                out[alias] = row
    return out


def _extract_topics(topic_row: Dict[str, Any], fallback_terms: Counter) -> List[Dict[str, Any]]:
    topics = []
    for item in _safe_list(topic_row.get("document_topics")):
        if not isinstance(item, dict):
            continue
        term = _normalize_term(item.get("term"))
        if not _is_informative_term(term):
            continue
        try:
            score = float(item.get("score") or 0.0)
        except Exception:
            score = 0.0
        topics.append({"term": term, "score": round(score, 6)})

    if topics:
        return topics[:14]

    if not fallback_terms:
        return []

    return [
        {"term": term, "score": round(float(freq), 6)}
        for term, freq in fallback_terms.most_common(14)
        if _is_informative_term(term)
    ]


def _prepare_docs(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_docs = _dedupe_docs(ctx.get("TOK_DOCS") or ctx.get("selected"))
    if not raw_docs:
        return []

    topic_index = _index_topic_rows(ctx)
    semantic_index = _build_semantic_sentence_index(ctx)
    cls_index = _build_classification_index(ctx)
    out: List[Dict[str, Any]] = []
    for i, doc in enumerate(raw_docs):
        doc_id = str(doc.get("doc_id") or "").strip() or None
        filename = str(doc.get("filename") or f"doc_{i+1}").strip()
        aliases = _doc_aliases(doc_id, filename, i)

        sentences = list(_iter_doc_sentences(doc, filename=filename, semantic_index=semantic_index))
        if len(sentences) > MAX_SENTENCES_PER_DOC:
            sentences = sentences[:MAX_SENTENCES_PER_DOC]

        term_counter: Counter = Counter()
        sentence_df: Counter = Counter()
        for sent in sentences:
            terms = sent.get("terms") or []
            term_counter.update(terms)
            sentence_df.update(set(terms))

        topic_row = {}
        for alias in aliases:
            if alias in topic_index:
                topic_row = topic_index[alias]
                break
        topics = _extract_topics(topic_row, term_counter)
        topic_terms = {str(t.get("term")) for t in topics if isinstance(t, dict) and t.get("term")}
        topic_scores = {
            str(t.get("term")): float(t.get("score") or 0.0)
            for t in topics
            if isinstance(t, dict) and t.get("term")
        }
        signal_terms: Set[str] = set(topic_terms)
        for alias in aliases:
            signal_terms.update(cls_index.get(alias) or set())

        out.append(
            {
                "doc_key": _doc_key(doc_id, filename, i),
                "aliases": aliases,
                "doc_id": doc_id,
                "filename": filename,
                "sentences": sentences,
                "term_counter": term_counter,
                "sentence_df": sentence_df,
                "topics": topics,
                "topic_terms": topic_terms,
                "topic_scores": topic_scores,
                "signal_terms": signal_terms,
            }
        )
    return out


def _score_informative_term(
    term: str,
    pair_sentence_df: Counter,
    total_sentences: int,
    shared_doc_topics: Set[str],
    shared_signal_terms: Set[str],
) -> float:
    df = max(1, int(pair_sentence_df.get(term) or 1))
    idf = math.log(1.0 + ((total_sentences + 1.0) / df))
    topic_bonus = 0.85 if term in shared_doc_topics else 0.0
    signal_bonus = 0.55 if term in shared_signal_terms else 0.0
    length_bonus = min(0.35, max(0.0, (len(term) - 4) * 0.03))
    return idf + topic_bonus + signal_bonus + length_bonus


def _sentence_match_score(
    sent_a: Dict[str, Any],
    sent_b: Dict[str, Any],
    shared_doc_topics: Set[str],
    shared_signal_terms: Set[str],
    pair_sentence_df: Counter,
    total_sentences: int,
) -> Tuple[float, List[str], List[str]]:
    terms_a = sent_a.get("terms_set") or set()
    terms_b = sent_b.get("terms_set") or set()
    shared = set(terms_a).intersection(terms_b)
    if not shared:
        return 0.0, [], []

    scored_terms: List[Tuple[str, float]] = []
    for term in shared:
        if not _is_informative_term(term):
            continue
        weight = _score_informative_term(
            term=term,
            pair_sentence_df=pair_sentence_df,
            total_sentences=total_sentences,
            shared_doc_topics=shared_doc_topics,
            shared_signal_terms=shared_signal_terms,
        )
        if weight < MIN_INFORMATIVE_TERM_SCORE:
            continue
        scored_terms.append((term, weight))

    if not scored_terms:
        return 0.0, [], []

    scored_terms.sort(key=lambda x: x[1], reverse=True)
    informative_terms = [term for term, _ in scored_terms[:MAX_SHARED_TERMS_PER_MATCH]]
    if not informative_terms:
        return 0.0, [], []

    if len(informative_terms) == 1:
        only = informative_terms[0]
        if only not in shared_doc_topics and only not in shared_signal_terms:
            return 0.0, [], []

    informative_topics = [term for term in informative_terms if term in shared_doc_topics]
    informative_signals = [term for term in informative_terms if term in shared_signal_terms]

    denom = float(len(terms_a) + len(terms_b) + 1)
    base = sum(weight for _, weight in scored_terms[:MAX_SHARED_TERMS_PER_MATCH]) / denom
    topic_boost = 1.0 + min(0.45, 0.17 * len(informative_topics))
    signal_boost = 1.0 + min(0.35, 0.12 * len(informative_signals))
    score = base * topic_boost * signal_boost
    return score, informative_terms, informative_topics


def _topic_examples_for_doc(doc: Dict[str, Any], term: str, limit: int = 2) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    doc_id = doc.get("doc_id")
    filename = doc.get("filename")
    for sent in _safe_list(doc.get("sentences")):
        if not isinstance(sent, dict):
            continue
        if term not in (sent.get("terms_set") or set()):
            continue
        out.append(
            {
                "document_id": doc_id,
                "filename": filename,
                "page_index": _safe_int(sent.get("page_index"), 0),
                "sent_index": _safe_int(sent.get("sent_index"), 0),
                "text_excerpt": _clip_text(str(sent.get("text") or "")),
            }
        )
        if len(out) >= limit:
            break
    return out


def _build_link(doc_a: Dict[str, Any], doc_b: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    shared_topics = set(doc_a.get("topic_terms") or set()).intersection(doc_b.get("topic_terms") or set())
    shared_signal_terms = set(doc_a.get("signal_terms") or set()).intersection(doc_b.get("signal_terms") or set())

    pair_sentence_df: Counter = Counter()
    sentences_a = [s for s in _safe_list(doc_a.get("sentences")) if isinstance(s, dict)]
    sentences_b = [s for s in _safe_list(doc_b.get("sentences")) if isinstance(s, dict)]
    for sent in sentences_a + sentences_b:
        pair_sentence_df.update(set(sent.get("terms_set") or set()))
    total_sentences = max(1, len(sentences_a) + len(sentences_b))

    scored_pairs = 0
    raw_matches: List[Dict[str, Any]] = []
    for sent_a in sentences_a:
        for sent_b in sentences_b:
            score, shared_terms, shared_topic_terms = _sentence_match_score(
                sent_a,
                sent_b,
                shared_doc_topics=shared_topics,
                shared_signal_terms=shared_signal_terms,
                pair_sentence_df=pair_sentence_df,
                total_sentences=total_sentences,
            )
            if score < MIN_SENTENCE_MATCH_SCORE:
                continue
            scored_pairs += 1
            raw_matches.append(
                {
                    "score": score,
                    "shared_terms": shared_terms,
                    "shared_topics": shared_topic_terms,
                    "phrase_a": {
                        "document_id": doc_a.get("doc_id"),
                        "filename": doc_a.get("filename"),
                        "page_index": _safe_int(sent_a.get("page_index"), 0),
                        "sent_index": _safe_int(sent_a.get("sent_index"), 0),
                        "text_excerpt": _clip_text(str(sent_a.get("text") or "")),
                    },
                    "phrase_b": {
                        "document_id": doc_b.get("doc_id"),
                        "filename": doc_b.get("filename"),
                        "page_index": _safe_int(sent_b.get("page_index"), 0),
                        "sent_index": _safe_int(sent_b.get("sent_index"), 0),
                        "text_excerpt": _clip_text(str(sent_b.get("text") or "")),
                    },
                }
            )

    raw_matches.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    selected_matches = raw_matches[:MAX_MATCHES_PER_LINK]

    best_sentence_score = float(selected_matches[0]["score"]) if selected_matches else 0.0
    shared_topics_count = len(shared_topics)
    shared_signals_count = len(shared_signal_terms)
    smallest_topics = max(1, min(len(doc_a.get("topic_terms") or set()), len(doc_b.get("topic_terms") or set())))
    topic_overlap = shared_topics_count / float(smallest_topics)
    smallest_signals = max(1, min(len(doc_a.get("signal_terms") or set()), len(doc_b.get("signal_terms") or set())))
    signal_overlap = shared_signals_count / float(smallest_signals)
    link_score = (0.45 * topic_overlap) + (0.35 * best_sentence_score) + (0.20 * signal_overlap)

    if shared_topics_count < MIN_SHARED_TOPICS and signal_overlap < 0.20 and best_sentence_score < 0.24:
        return {}, scored_pairs
    if link_score < MIN_LINK_SCORE:
        return {}, scored_pairs

    shared_topics_scored = []
    for term in sorted(shared_topics):
        score_a = float((doc_a.get("topic_scores") or {}).get(term, 0.0))
        score_b = float((doc_b.get("topic_scores") or {}).get(term, 0.0))
        shared_topics_scored.append(
            {
                "term": term,
                "score": round(min(score_a, score_b), 6),
                "doc_a_examples": _topic_examples_for_doc(doc_a, term, limit=2),
                "doc_b_examples": _topic_examples_for_doc(doc_b, term, limit=2),
            }
        )
    shared_topics_scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    link = {
        "score": round(link_score, 6),
        "doc_a": {
            "document_id": doc_a.get("doc_id"),
            "filename": doc_a.get("filename"),
        },
        "doc_b": {
            "document_id": doc_b.get("doc_id"),
            "filename": doc_b.get("filename"),
        },
        "shared_topics": shared_topics_scored[:12],
        "audit": {
            "sentence_matches_count": len(selected_matches),
            "matches": [
                {
                    "score": round(float(m.get("score") or 0.0), 6),
                    "shared_terms": _safe_list(m.get("shared_terms"))[:10],
                    "shared_topics": _safe_list(m.get("shared_topics"))[:8],
                    "phrase_a": m.get("phrase_a"),
                    "phrase_b": m.get("phrase_b"),
                }
                for m in selected_matches
            ],
        },
    }
    return link, scored_pairs


def _compute_links(prepared_docs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    links: List[Dict[str, Any]] = []
    sentence_pairs_scored = 0

    for i in range(len(prepared_docs)):
        for j in range(i + 1, len(prepared_docs)):
            link, scored = _build_link(prepared_docs[i], prepared_docs[j])
            sentence_pairs_scored += scored
            if not link:
                continue
            link["link_id"] = f"link-{len(links) + 1}"
            links.append(link)

    links.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    for idx, link in enumerate(links, start=1):
        link["link_id"] = f"link-{idx}"
    return links, sentence_pairs_scored


def _build_doc_links_index(prepared_docs: List[Dict[str, Any]], links: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    index: Dict[str, Set[str]] = {}
    doc_aliases_by_name: Dict[str, List[str]] = {}
    for i, doc in enumerate(prepared_docs):
        filename = str(doc.get("filename") or "")
        doc_aliases_by_name[filename] = _doc_aliases(doc.get("doc_id"), filename, i)

    for link in links:
        link_id = str(link.get("link_id") or "")
        if not link_id:
            continue
        for side in ("doc_a", "doc_b"):
            doc_ref = link.get(side) if isinstance(link.get(side), dict) else {}
            filename = str(doc_ref.get("filename") or "")
            aliases = doc_aliases_by_name.get(filename) or _doc_aliases(doc_ref.get("document_id"), filename, 0)
            for alias in aliases:
                index.setdefault(alias, set()).add(link_id)

    return {k: sorted(v) for k, v in index.items()}


def run(ctx: Dict[str, Any]) -> Dict[str, Any]:
    prepared_docs = _prepare_docs(ctx)
    pairs_evaluated = int((len(prepared_docs) * (len(prepared_docs) - 1)) / 2)
    links, sentence_pairs_scored = _compute_links(prepared_docs)
    doc_links = _build_doc_links_index(prepared_docs, links)

    analysis = {
        "method": "topic-sentence-audit-v1",
        "generated_at": _iso_now(),
        "documents_analyzed": len(prepared_docs),
        "pairs_evaluated": pairs_evaluated,
        "sentence_pairs_scored": sentence_pairs_scored,
        "links_count": len(links),
        "links": links,
    }

    ctx["INTERDOC_LINKS"] = links
    ctx["INTERDOC_DOC_LINKS"] = doc_links
    ctx["INTERDOC_ANALYSIS"] = analysis

    print(
        "[interdoc-link] "
        f"docs={len(prepared_docs)} | pairs={pairs_evaluated} | links={len(links)} | "
        f"sentence_pairs_scored={sentence_pairs_scored}"
    )
    for link in links[:8]:
        a = link.get("doc_a") if isinstance(link.get("doc_a"), dict) else {}
        b = link.get("doc_b") if isinstance(link.get("doc_b"), dict) else {}
        shared = [str(x.get("term")) for x in _safe_list(link.get("shared_topics"))[:4] if isinstance(x, dict)]
        print(
            "[interdoc-link] "
            f"{a.get('filename')} <-> {b.get('filename')} | score={link.get('score')} | "
            f"shared_topics={shared} | sentence_matches={_safe_int((link.get('audit') or {}).get('sentence_matches_count'), 0)}"
        )
    return analysis


_CTX = globals()
run(_CTX)
