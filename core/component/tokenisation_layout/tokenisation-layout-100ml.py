from __future__ import annotations

import hashlib
import math
import re
import runpy
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

BASE_SCRIPT = Path(__file__).resolve().with_name("tokenisation-layout.py")
TOKEN_RE = re.compile(r"[A-Za-z0-9_\u00C0-\u024F\u0600-\u06FF]+", re.UNICODE)
Vector = List[float]

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "will", "shall", "must",
    "dans", "avec", "pour", "par", "les", "des", "une", "sur", "aux", "est", "sont", "sera", "etre", "ce",
    "de", "du", "la", "le", "un", "en", "et", "ou", "au", "aux",
    "في", "من", "على", "الى", "إلى", "عن", "مع", "و", "او", "أو", "هذا", "هذه",
    "qui", "que", "quoi", "dont", "where", "when", "what", "which", "who", "whom", "whose",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "i", "you", "he", "she", "it", "we", "they",
    "plus", "moins", "tres", "très",
    "page", "pages", "document", "documents", "corpus", "test", "tests", "positive", "negative",
    "regex", "info", "valeur", "reference", "references", "note", "notes", "file", "files",
    "section", "annexe", "annex", "article", "chapitre",
}

_TOPIC_NOISE_TERMS = {
    "http", "https", "www", "com", "net", "org", "pdf", "doc", "image", "text",
    "unknown", "none", "null", "true", "false", "nan",
}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _norm_token(token: str) -> str:
    return str(token or "").strip().lower()


def _tokenize(text: str) -> List[str]:
    return [_norm_token(t) for t in TOKEN_RE.findall(str(text or "")) if _norm_token(t)]


def _norm_topic_term(term: str) -> str:
    val = _norm_token(term)
    if not val:
        return ""
    val = unicodedata.normalize("NFKD", val)
    val = "".join(ch for ch in val if not unicodedata.combining(ch))
    val = val.replace("’", "'").replace("`", "'")
    return val


def _is_topic_candidate(term: str) -> bool:
    t = _norm_topic_term(term)
    if not t:
        return False
    if t in _STOPWORDS or t in _TOPIC_NOISE_TERMS:
        return False
    if len(t) < 3 or len(t) > 48:
        return False
    if t.isdigit():
        return False
    if t.count("_") > 1:
        return False
    alpha_count = sum(1 for ch in t if ch.isalpha())
    if alpha_count == 0:
        return False
    digit_count = sum(1 for ch in t if ch.isdigit())
    if digit_count > alpha_count:
        return False
    return True


def _tokens_to_topic_terms(tokens: List[str], max_ngram: int = 2) -> List[str]:
    base = [_norm_topic_term(t) for t in tokens]
    base = [t for t in base if _is_topic_candidate(t)]
    if not base:
        return []
    out: List[str] = list(base)
    if max_ngram >= 2:
        for i in range(len(base) - 1):
            a, b = base[i], base[i + 1]
            if a in _STOPWORDS or b in _STOPWORDS:
                continue
            if _is_topic_candidate(a) and _is_topic_candidate(b):
                out.append(f"{a} {b}")
    return out


def _score_term_quality(term: str) -> float:
    if not term:
        return 0.0
    words = term.split()
    if len(term) > 42:
        return 0.7
    if len(words) >= 2:
        return 1.2
    return 1.0


def _prune_topic_redundancy(scored_terms: List[Tuple[str, float]], max_topics: int) -> List[Tuple[str, float]]:
    kept: List[Tuple[str, float]] = []
    kept_set = set()
    for term, score in scored_terms:
        if term in kept_set:
            continue
        if " " not in term:
            skip = False
            for kterm, _ in kept:
                if " " in kterm and term in kterm.split():
                    skip = True
                    break
            if skip:
                continue
        kept.append((term, score))
        kept_set.add(term)
        if len(kept) >= max_topics:
            break
    return kept


def _extract_terms_from_keyword_item(item: Any) -> List[str]:
    text = ""
    if isinstance(item, dict):
        text = str(item.get("keyword") or item.get("value") or item.get("term") or "")
    else:
        text = str(item or "")
    if not text:
        return []
    text = re.sub(r"\(x\d+\)$", "", text.strip(), flags=re.IGNORECASE)
    return [_norm_topic_term(t) for t in _tokenize(text) if _is_topic_candidate(t)]


def _build_topic_boost_terms(ctx: Dict[str, Any], doc_id: Any, filename: str) -> Dict[str, float]:
    results = ctx.get("RESULTS") or []
    if not isinstance(results, list):
        return {}

    selected: Dict[str, Any] = {}
    sid = str(doc_id or "").strip()
    sfn = str(filename or "").strip()
    for row in results:
        if not isinstance(row, dict):
            continue
        if sid and str(row.get("doc_id") or "").strip() == sid:
            selected = row
            break
        if sfn and str(row.get("filename") or "").strip() == sfn:
            selected = row
            break
    if not selected:
        return {}

    boost: Dict[str, float] = {}

    def _apply(items: Any, factor: float) -> None:
        if not isinstance(items, list):
            return
        for it in items:
            for term in _extract_terms_from_keyword_item(it):
                boost[term] = max(boost.get(term, 1.0), factor)

    doc_type = str(selected.get("doc_type") or "").strip().lower()
    if doc_type:
        for term in [_norm_topic_term(t) for t in _tokenize(doc_type)]:
            if _is_topic_candidate(term):
                boost[term] = max(boost.get(term, 1.0), 1.25)

    kw = selected.get("keyword_matches")
    if isinstance(kw, dict):
        _apply(kw.get("strong"), 1.65)
        _apply(kw.get("medium"), 1.35)
        _apply(kw.get("weak"), 1.12)
        _apply(kw.get("negative"), 0.8)
        _apply(kw.get("strong_negative"), 0.65)
        _apply(kw.get("anti_confusion_hits"), 0.72)

    return boost


def _hash_index_sign(value: str, dim: int) -> Tuple[int, float]:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
    idx = int.from_bytes(digest[:8], "big") % dim
    sign = 1.0 if (digest[8] & 1) == 0 else -1.0
    return idx, sign


def _vector_norm(vec: Vector) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def _hash_text_vector(text: str, dim: int) -> Vector:
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        tokens = [str(text or "").strip().lower()]
    for token in tokens:
        if not token:
            continue
        idx, sign = _hash_index_sign(token, dim)
        vec[idx] += sign
    norm = _vector_norm(vec)
    if norm > 0.0:
        vec = [float(x) / norm for x in vec]
    return vec


def _mean_vectors(vectors: List[Vector], dim: int) -> Vector:
    if not vectors:
        return [0.0] * dim
    out = [0.0] * dim
    for v in vectors:
        limit = min(dim, len(v))
        for i in range(limit):
            out[i] += float(v[i])
    out = [x / float(len(vectors)) for x in out]
    norm = _vector_norm(out)
    if norm > 0.0:
        out = [float(x) / norm for x in out]
    return out


def _to_list(vec: Vector, precision: int = 6) -> List[float]:
    return [round(float(x), precision) for x in vec]


def _iter_doc_chunks(doc: Dict[str, Any]) -> Iterable[Tuple[int, int, str, str]]:
    for page in doc.get("pages") or []:
        if not isinstance(page, dict):
            continue
        page_index = _safe_int(page.get("page_index"), 1)
        lang = str(page.get("lang") or "")
        sents = page.get("sentences_layout")
        if isinstance(sents, list) and sents:
            for sent_index, sent in enumerate(sents):
                if not isinstance(sent, dict):
                    continue
                text = str(sent.get("text") or "")
                if text.strip():
                    yield page_index, sent_index, text, lang
            continue

        page_text = str(page.get("page_text") or page.get("text") or "")
        if page_text.strip():
            yield page_index, 0, page_text, lang


def _extract_topics_from_chunks(
    tokenized_chunks: List[List[str]],
    max_topics: int = 12,
    boost_terms: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    if not tokenized_chunks:
        return []

    filtered_chunks: List[List[str]] = [_tokens_to_topic_terms(chunk, max_ngram=2) for chunk in tokenized_chunks]
    filtered_chunks = [c for c in filtered_chunks if c]
    if not filtered_chunks:
        return []

    tf = Counter()
    df = Counter()
    for terms in filtered_chunks:
        tf.update(terms)
        df.update(set(terms))

    total_chunks = max(1, len(filtered_chunks))
    scored: List[Tuple[str, float]] = []
    boosts = boost_terms or {}
    for term, freq in tf.items():
        term_df = max(1, int(df.get(term) or 1))
        idf = math.log(1.0 + (total_chunks + 1.0) / term_df)
        boost = float(boosts.get(term, 1.0))
        quality = _score_term_quality(term)
        score = float(freq) * idf * quality * boost
        scored.append((term, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    pruned = _prune_topic_redundancy(scored, max_topics)
    return [{"term": term, "score": round(score, 6)} for term, score in pruned]


def _extract_chunk_topics(
    tokens: List[str],
    max_topics: int = 3,
    doc_df: Counter | None = None,
    doc_chunks_count: int = 1,
    boost_terms: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    terms = _tokens_to_topic_terms(tokens, max_ngram=2)
    if not terms:
        return []
    counts = Counter(terms)
    total = float(sum(counts.values()) or 1.0)
    scored: List[Tuple[str, float]] = []
    boosts = boost_terms or {}
    total_chunks = max(1, int(doc_chunks_count))
    for term, freq in counts.items():
        idf = 1.0
        if isinstance(doc_df, Counter):
            term_df = max(1, int(doc_df.get(term) or 1))
            idf = math.log(1.0 + (total_chunks + 1.0) / term_df)
        tf_norm = float(freq) / total
        boost = float(boosts.get(term, 1.0))
        quality = _score_term_quality(term)
        scored.append((term, tf_norm * idf * quality * boost))
    scored.sort(key=lambda x: x[1], reverse=True)
    pruned = _prune_topic_redundancy(scored, max_topics)
    return [{"term": term, "score": round(score, 6)} for term, score in pruned]


def _doc_key(doc_id: Any, filename: Any) -> str:
    sid = str(doc_id or "").strip()
    if sid:
        return f"id:{sid}"
    sfn = str(filename or "").strip().lower()
    return f"fn:{sfn}"


class _TransformerEmbedder:
    def __init__(self, model_name: str, max_length: int, batch_size: int, fallback_dim: int):
        self.model_name = model_name
        self.max_length = max(16, max_length)
        self.batch_size = max(1, batch_size)
        self.dim = max(64, fallback_dim)
        self.backend = "hash-fallback"
        self.error: str | None = None

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            hidden_size = getattr(getattr(self._model, "config", None), "hidden_size", None)
            if hidden_size:
                self.dim = int(hidden_size)
            self.backend = f"transformer:{self.model_name}"
        except Exception as exc:
            self.error = str(exc)
            self._torch = None
            self._tokenizer = None
            self._model = None

    def embed_texts(self, texts: List[str]) -> List[Vector]:
        clean_texts = [str(t or "") for t in texts]
        if not clean_texts:
            return []

        if self._torch is None or self._tokenizer is None or self._model is None:
            return [_hash_text_vector(text, self.dim) for text in clean_texts]

        torch = self._torch
        out: List[Vector] = []

        try:
            with torch.inference_mode():
                for i in range(0, len(clean_texts), self.batch_size):
                    batch = clean_texts[i:i + self.batch_size]
                    enc = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    outputs = self._model(**enc)
                    hidden = getattr(outputs, "last_hidden_state", None)
                    if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
                        hidden = outputs[0]
                    if hidden is None:
                        for text in batch:
                            out.append(_hash_text_vector(text, self.dim))
                        continue

                    mask = enc.get("attention_mask")
                    if mask is None:
                        mask = torch.ones(hidden.shape[:2], device=hidden.device)
                    mask = mask.unsqueeze(-1).type_as(hidden)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    pooled = pooled / pooled.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)

                    rows = pooled.detach().cpu().tolist()
                    for row in rows:
                        out.append([float(x) for x in row])
        except Exception as exc:
            self.error = str(exc)
            self.backend = "hash-fallback"
            return [_hash_text_vector(text, self.dim) for text in clean_texts]

        if len(out) < len(clean_texts):
            out.extend(_hash_text_vector(clean_texts[i], self.dim) for i in range(len(out), len(clean_texts)))
        return out


def _run_base_tokenisation(ctx: Dict[str, Any]) -> None:
    result = runpy.run_path(str(BASE_SCRIPT), run_name="__main__", init_globals=ctx)
    if isinstance(result, dict):
        ctx.update(result)


def _augment_with_ml100(ctx: Dict[str, Any]) -> None:
    tok_docs = ctx.get("TOK_DOCS") or ctx.get("selected") or []
    if not isinstance(tok_docs, list):
        tok_docs = []

    model_name = str(ctx.get("ML100_MODEL_NAME") or "xlm-roberta-base")
    max_length = _safe_int(ctx.get("ML100_MAX_LENGTH"), 256)
    batch_size = _safe_int(ctx.get("ML100_BATCH_SIZE"), 8)
    fallback_dim = _safe_int(ctx.get("ML100_HASH_FALLBACK_DIM"), 384)

    embedder = _TransformerEmbedder(model_name, max_length, batch_size, fallback_dim)
    dim = int(embedder.dim)

    chunk_rows: List[Dict[str, Any]] = []
    topics_rows: List[Dict[str, Any]] = []
    nlp_analyses: List[Dict[str, Any]] = []
    nlp_tokens: List[Dict[str, Any]] = []
    lang_counter: Counter = Counter()

    doc_meta: List[Dict[str, Any]] = []
    doc_key_to_chunk_vecs: Dict[str, List[Vector]] = {}
    doc_topic_stats: Dict[str, Dict[str, Any]] = {}

    for doc in tok_docs:
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id")
        filename = str(doc.get("filename") or "")
        key = _doc_key(doc_id, filename)
        doc_meta.append({"doc_id": doc_id, "filename": filename, "doc_key": key})
        doc_boost_terms = _build_topic_boost_terms(ctx, doc_id, filename)

        tokenized_chunks: List[List[str]] = []

        for page_index, sent_index, text, lang in _iter_doc_chunks(doc):
            tokens = _tokenize(text)
            if not tokens:
                continue
            lang_counter[lang or "unknown"] += 1
            tokenized_chunks.append(tokens)

            chunk_rows.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "doc_key": key,
                    "page_index": page_index,
                    "sent_index": sent_index,
                    "lang": lang or None,
                    "token_count": len(tokens),
                    "text_preview": text[:240],
                    "tokens": tokens,
                    "_raw_text": text,
                }
            )

            lemmas = [t.lower() for t in tokens]
            pos = ["UNK"] * len(tokens)
            ner = ["O"] * len(tokens)
            nlp_analyses.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "doc": filename,
                    "page_index": page_index,
                    "sent_index": sent_index,
                    "lang": lang or None,
                    "text": text,
                    "tokens": tokens,
                    "lemmas": lemmas,
                    "pos": pos,
                    "ner_labels": ner,
                    "entities": [],
                }
            )
            for tok_index, tok in enumerate(tokens):
                nlp_tokens.append(
                    {
                        "doc_id": doc_id,
                        "filename": filename,
                        "page_index": page_index,
                        "sent_index": sent_index,
                        "tok_index": tok_index,
                        "token": tok,
                        "lemma": lemmas[tok_index],
                        "pos": "UNK",
                        "ner": "O",
                        "lang": lang or None,
                    }
                )

        doc_df: Counter = Counter()
        for chunk_tokens in tokenized_chunks:
            terms = set(_tokens_to_topic_terms(chunk_tokens, max_ngram=2))
            if terms:
                doc_df.update(terms)

        doc_topic_stats[key] = {
            "doc_df": doc_df,
            "doc_chunks_count": max(1, len(tokenized_chunks)),
            "boost_terms": doc_boost_terms,
        }

        doc_topics = _extract_topics_from_chunks(tokenized_chunks, boost_terms=doc_boost_terms)
        topics_rows.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "doc_key": key,
                "document_topics": doc_topics,
                "document_primary_topics": [
                    x.get("term") for x in doc_topics[:2] if isinstance(x, dict) and x.get("term")
                ],
            }
        )

    chunk_texts = [str(row.get("_raw_text") or "") for row in chunk_rows]
    chunk_vecs = embedder.embed_texts(chunk_texts)

    chunk_vectors: List[Dict[str, Any]] = []
    for i, row in enumerate(chunk_rows):
        vec = chunk_vecs[i] if i < len(chunk_vecs) else [0.0] * dim
        key = str(row.get("doc_key") or "")
        doc_key_to_chunk_vecs.setdefault(key, []).append(vec)
        stats = doc_topic_stats.get(key) or {}
        chunk_topics = _extract_chunk_topics(
            row.get("tokens") or [],
            max_topics=3,
            doc_df=stats.get("doc_df"),
            doc_chunks_count=int(stats.get("doc_chunks_count") or 1),
            boost_terms=stats.get("boost_terms") if isinstance(stats.get("boost_terms"), dict) else None,
        )

        out_row = dict(row)
        out_row.pop("_raw_text", None)
        out_row.pop("tokens", None)
        out_row["chunk_primary_topic"] = chunk_topics[0]["term"] if chunk_topics else None
        out_row["chunk_topics"] = chunk_topics
        out_row["vector"] = _to_list(vec)
        chunk_vectors.append(out_row)

    doc_vectors: List[Dict[str, Any]] = []
    for meta in doc_meta:
        key = str(meta.get("doc_key") or "")
        doc_chunk_vecs = doc_key_to_chunk_vecs.get(key) or []
        doc_vec = _mean_vectors(doc_chunk_vecs, dim)
        doc_vectors.append(
            {
                "doc_id": meta.get("doc_id"),
                "filename": meta.get("filename"),
                "doc_key": key,
                "dim": dim,
                "chunk_count": len(doc_chunk_vecs),
                "vector": _to_list(doc_vec),
            }
        )

    detected_languages = [lang for lang, _ in lang_counter.most_common()]
    dominant_lang = detected_languages[0] if detected_languages else None

    ctx["ML100_EMBEDDING_METHOD"] = "transformer-mean-pooling"
    ctx["ML100_EMBEDDING_BACKEND"] = embedder.backend
    ctx["ML100_MODEL_NAME"] = model_name
    ctx["ML100_VECTOR_DIM"] = dim
    ctx["ML100_DOC_VECTORS"] = doc_vectors
    ctx["ML100_CHUNK_VECTORS"] = chunk_vectors
    ctx["ML100_WORD_VECTORS"] = []
    ctx["ML100_TOPICS"] = topics_rows

    # Keep NLP contract for ES summary/full even without grammar component.
    ctx["NLP_ANALYSES"] = nlp_analyses
    ctx["NLP_SENTENCES"] = nlp_analyses
    ctx["NLP_ENTITIES"] = []
    ctx["NLP_TOKENS"] = nlp_tokens
    ctx["NLP_POS"] = nlp_tokens
    ctx["NLP_LEMMA"] = nlp_tokens
    ctx["NLP_LANGUAGE"] = dominant_lang
    ctx["NLP_LANGUAGE_STATS"] = {lang: int(cnt) for lang, cnt in lang_counter.items()}
    ctx["DETECTED_LANGUAGES"] = detected_languages

    if embedder.error:
        print(
            "[tokenisation-100ml] transformer indisponible, fallback hash | "
            f"reason={embedder.error[:180]}"
        )

    for row in topics_rows:
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename") or row.get("doc_id") or "unknown")
        document_primary_topics = [str(x) for x in (row.get("document_primary_topics") or []) if x]
        top_topics = [
            str(item.get("term"))
            for item in (row.get("document_topics") or [])[:5]
            if isinstance(item, dict) and item.get("term")
        ]
        print(
            f"[topic-doc-100] {filename} | document_primary_topics={document_primary_topics} | "
            f"document_top_topics={top_topics}"
        )

    print(
        "[tokenisation-100ml] "
        f"docs={len(doc_vectors)} | chunks={len(chunk_vectors)} | words=0 | "
        f"topics={sum(len(x.get('document_topics') or []) for x in topics_rows)} | "
        f"nlp_rows={len(nlp_analyses)} | vector_dim={dim} | backend={embedder.backend}"
    )


_CTX = globals()
_run_base_tokenisation(_CTX)
_augment_with_ml100(_CTX)
