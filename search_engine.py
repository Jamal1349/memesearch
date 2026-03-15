import math
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Any
from typing import Optional
import hashlib
import faiss
import numpy as np
from CLIP import STClipVectorizer

import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from app_config import AppConfig
from storage import load_json_file

try:
    import pymorphy3
except Exception:
    pymorphy3 = None

_MORPH = pymorphy3.MorphAnalyzer() if pymorphy3 is not None else None


RUSSIAN_STOPWORDS = {
    "и", "в", "во", "на", "но", "а", "или", "что", "это", "как", "к", "ко", "по",
    "из", "у", "за", "от", "до", "не", "ни", "же", "ли", "бы", "то", "для", "с",
    "со", "о", "об", "под", "над", "при", "мы", "вы", "он", "она", "они", "я",
    "ты", "его", "ее", "их", "мне", "тебе", "нам", "вас", "тут", "там",
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "with", "for", "is", "are", "was", "were", "be",
}

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
CYR_RE = re.compile(r"[а-яё]+$", re.IGNORECASE)


def normalize_query_key(query: str) -> str:
    return (query or "").strip().lower()


def history_query_key(query: str) -> str:
    return normalize_query_key(query)[:200]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def summarize_text(text: str, max_tokens: int = 40) -> str:
    tokens = tokenize(text)
    return " ".join(tokens[:max_tokens])


def coerce_text_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [coerce_text_value(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


@lru_cache(maxsize=50000)
def _lemmatize_token(token: str) -> str:
    if not token:
        return token
    if pymorphy3 is None:
        return token
    return _MORPH.parse(token)[0].normal_form


def normalize_tokens(text: str, max_tokens: int | None = None) -> list[str]:
    tokens = tokenize(text)
    normalized: list[str] = []
    for token in tokens:
        lemma = _lemmatize_token(token)
        if len(lemma) < 2 or lemma in RUSSIAN_STOPWORDS or lemma in ENGLISH_STOPWORDS:
            continue
        normalized.append(lemma)
        if max_tokens is not None and len(normalized) >= max_tokens:
            break
    return normalized


def build_query_variants(query: str) -> list[str]:
    query = (query or "").strip()
    if not query:
        return []

    normalized_tokens = normalize_tokens(query, max_tokens=12)
    variants = [query]

    if normalized_tokens:
        lemma_query = " ".join(normalized_tokens)
        variants.append(lemma_query)
        variants.append("фото " + lemma_query)
        variants.append("картинка " + lemma_query)

    unique: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


class BM25Index:
    def __init__(self, documents: list[list[str]], doc_ids: list[int], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.doc_ids = doc_ids
        self.k1 = k1
        self.b = b
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        total_len = 0
        for tokens in self.documents:
            counts = Counter(tokens)
            self.term_freqs.append(counts)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len
            for token in counts:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        self.avgdl = total_len / len(self.documents) if self.documents else 0.0

    def score(self, query_tokens: list[str], topn: int = 100) -> list[tuple[int, float]]:
        if not query_tokens or not self.documents:
            return []

        query_terms = Counter(query_tokens)
        scored: list[tuple[int, float]] = []
        doc_count = len(self.documents)

        for doc_idx, term_freq in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lens[doc_idx] or 1
            norm = self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl)) if self.avgdl else self.k1

            for term, qtf in query_terms.items():
                freq = term_freq.get(term, 0)
                if not freq:
                    continue
                doc_freq = self.doc_freqs.get(term, 0)
                idf = math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
                score += idf * ((freq * (self.k1 + 1)) / (freq + norm)) * qtf

            if score > 0:
                scored.append((self.doc_ids[doc_idx], score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:topn]


class SearchEngine:
    def __init__(self, config: AppConfig, logger):
        self.config = config
        self.logger = logger
        self.clip_vec = None
        self.clip_index = None
        self.clip_meta = None
        self.df = None
        self.bm25_index = None
        self.search_sessions: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        self._load_clip()
        self._load_dataset()
        self._init_text_search()

    def _load_clip(self) -> None:
        self.logger.info(
            "CLIP paths: model=%s index=%s meta=%s",
            self.config.clip_model_path,
            self.config.clip_index_path,
            self.config.clip_meta_path,
        )
        if not (os.path.exists(self.config.clip_index_path) and os.path.exists(self.config.clip_meta_path)):
            self.logger.warning("CLIP index not found, using lexical search fallback")
            return

        try:
            self.clip_vec = STClipVectorizer(model_name=self.config.clip_model_path)
            self.clip_index = faiss.read_index(self.config.clip_index_path)
            self.clip_meta = np.load(self.config.clip_meta_path)
            self.logger.info("CLIP ready: ntotal=%s", self.clip_index.ntotal)
        except Exception:
            self.clip_vec = None
            self.clip_index = None
            self.clip_meta = None
            self.logger.exception("Failed to load CLIP. Falling back to BM25-only mode.")

    def _load_dataset(self) -> None:
        if os.path.exists(self.config.dataset_path):
            self.logger.info("Loading dataset from disk: %s", self.config.dataset_path)
            dataset = load_from_disk(self.config.dataset_path)
            if isinstance(dataset, DatasetDict):
                split_names = list(dataset.keys())
                self.logger.info("Combining dataset splits for search: %s", ", ".join(split_names))
                dataset = concatenate_datasets([dataset[split_name] for split_name in split_names])
        else:
            self.logger.info(
                "Loading dataset from HuggingFace: %s [%s]",
                self.config.dataset_name,
                self.config.dataset_split,
            )
            dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        self.df = dataset.to_pandas()
        if "original_id" in self.df.columns:
            self.df["original_id"] = self.df["original_id"].astype(int)
            self.df = self.df.set_index("original_id", drop=False)
        self.df["source"] = "hf"
        self._merge_local_memes()
        self.logger.info("Loaded %s memes", len(self.df))

    def _merge_local_memes(self) -> None:
        local_items = load_json_file(self.config.local_memes_path, [])
        if not local_items:
            return

        rows = []
        next_idx = int(self.df.index.max()) + 1 if len(self.df.index) else 0
        for offset, item in enumerate(local_items):
            image_path = item.get("image_path")
            if not image_path:
                continue
            rows.append(
                {
                    "description": item.get("description", ""),
                    "alt": item.get("alt", ""),
                    "query_phrases": item.get("query_phrases", ""),
                    "keywords": item.get("keywords", ""),
                    "ocr_text": item.get("ocr_text", ""),
                    "image": {"path": image_path},
                    "source": "local",
                    "_index": next_idx + offset,
                }
            )

        if not rows:
            return

        local_df = pd.DataFrame(rows).set_index("_index")
        self.df = pd.concat([self.df, local_df], axis=0)
        self.logger.info("Merged %s local memes", len(local_df))

    def _init_text_search(self) -> None:
        if self.df is None or len(self.df) == 0:
            self.bm25_index = None
            return
        self.df["search_text"] = self.df.apply(self._build_search_text, axis=1)
        self.df["search_terms"] = self.df["search_text"].apply(lambda text: normalize_tokens(text))
        documents = self.df["search_terms"].tolist()
        doc_ids = [int(idx) for idx in self.df.index.tolist()]
        self.bm25_index = BM25Index(documents, doc_ids)
        if pymorphy3 is None:
            self.logger.warning("pymorphy3 is not available, BM25 uses raw tokens without lemmatization")
        else:
            self.logger.info("BM25 uses lemmatized Russian tokens")

    def add_local_meme(self, image_path: str, description: str, alt: str = "") -> int:
        next_idx = int(self.df.index.max()) + 1 if self.df is not None and len(self.df.index) else 0
        row = pd.DataFrame(
            [
                {
                    "description": description,
                    "alt": alt,
                    "query_phrases": "",
                    "keywords": "",
                    "ocr_text": "",
                    "image": {"path": image_path},
                    "source": "local",
                }
            ],
            index=[next_idx],
        )
        self.df = pd.concat([self.df, row], axis=0)
        self._init_text_search()
        return next_idx

    def delete_local_meme(self, idx: int) -> dict | None:
        if self.df is None or idx not in self.df.index:
            return None

        row = self.df.loc[idx]
        if coerce_text_value(row.get("source")) != "local":
            return None

        record = row.to_dict()
        self.df = self.df.drop(index=idx)
        self._init_text_search()
        return record

    def _build_search_text(self, row) -> str:
        query_text = summarize_text(coerce_text_value(row.get("query_phrases")), max_tokens=48)
        keywords_text = summarize_text(coerce_text_value(row.get("keywords")), max_tokens=32)
        ocr_text = summarize_text(coerce_text_value(row.get("ocr_text")), max_tokens=80)
        description_text = summarize_text(coerce_text_value(row.get("description")), max_tokens=64)
        alt_text = summarize_text(coerce_text_value(row.get("alt")), max_tokens=32)
        parts = [
            query_text,
            query_text,
            keywords_text,
            keywords_text,
            ocr_text,
            description_text,
            alt_text,
        ]
        return " ".join(part.strip() for part in parts if part and part.strip()).lower()

    def total(self) -> int:
        return 0 if self.df is None else len(self.df)

    def row(self, idx: int):
        if self.df is None:
            raise KeyError("Dataset is not loaded")
        if idx in self.df.index:
            return self.df.loc[idx]
        if 0 <= idx < len(self.df):
            return self.df.iloc[idx]
        raise KeyError(f"Meme row not found: {idx}")

    def random_idx(self) -> int:
        return int(self.df.sample(1).iloc[0].name)

    def title_for_idx(self, idx: int) -> str:
        row = self.row(idx)
        return (
            coerce_text_value(row.get("query_phrases"))
            or coerce_text_value(row.get("description"))
            or coerce_text_value(row.get("alt"))
            or f"Мем #{idx}"
        )

    def latest_local_indices(self, limit: int = 10) -> list[int]:
        if self.df is None or len(self.df) == 0:
            return []
        local_df = self.df[self.df["source"] == "local"]
        if local_df.empty:
            return []
        return [int(idx) for idx in local_df.tail(max(limit, 0)).index.tolist()[::-1]]

    def _new_search_token(self, user_id: int, query: str) -> str:
        seed = f"{user_id}:{query}:{len(self.search_sessions)}".encode("utf-8")
        return hashlib.sha1(seed).hexdigest()[:16]

    def _trim_search_sessions(self) -> None:
        while len(self.search_sessions) > self.config.query_token_limit:
            oldest_token = next(iter(self.search_sessions))
            self.search_sessions.pop(oldest_token, None)

    def start_search_session(self, query: str, user_id: int, page_size: int = 5) -> tuple[str, list[int], bool]:
        session_limit = max(page_size * self.config.query_history_max_queries, self.config.query_history_max_results)
        ranked_results = self.search(query, limit=session_limit)
        token = self._new_search_token(user_id, query)
        self.search_sessions[token] = {
            "user_id": user_id,
            "query": query,
            "results": ranked_results,
            "cursor": 0,
        }
        self._trim_search_sessions()
        batch = self.next_search_results(token, user_id, page_size)
        return token, batch, self.search_session_has_more(token)

    def next_search_results(self, token: str, user_id: int, page_size: int = 5) -> list[int]:
        session = self.search_sessions.get(token)
        if not session:
            return []
        if int(session.get("user_id", -1)) != user_id:
            return []
        results = list(session.get("results", []))
        cursor = int(session.get("cursor", 0))
        batch = results[cursor : cursor + page_size]
        session["cursor"] = cursor + len(batch)
        return batch

    def search_session_has_more(self, token: str) -> bool:
        session = self.search_sessions.get(token)
        if not session:
            return False
        results = list(session.get("results", []))
        cursor = int(session.get("cursor", 0))
        return cursor < len(results)

    def search_session_query(self, token: str) -> str:
        session = self.search_sessions.get(token)
        if not session:
            return ""
        return str(session.get("query") or "")

    def search_keyword(self, query: str, limit: int = 10) -> list[int]:
        if self.df is None or self.bm25_index is None:
            return []
        query_key = history_query_key(query)
        if not query_key or len(query_key) < 2:
            return []

        query_tokens = normalize_tokens(query_key, max_tokens=12)
        if not query_tokens:
            return []
        scored_candidates = self.bm25_index.score(query_tokens, topn=max(limit * 30, 300))
        if not scored_candidates:
            return []
        if scored_candidates[0][1] < self.config.min_bm25_score:
            self.logger.info(
                "Rejecting BM25 results for query=%r: top_score=%.3f < %.3f",
                query,
                scored_candidates[0][1],
                self.config.min_bm25_score,
            )
            return []

        return [
            idx
            for idx, score in scored_candidates
            if score >= self.config.min_bm25_score
        ][:limit]

    def search_clip(self, query: str, limit: int = 10) -> list[int]:
        if self.clip_index is None or self.clip_vec is None:
            return []

        query_text = (query or "").strip()
        if not query_text:
            return []

        variants = build_query_variants(query_text)
        if not variants:
            return []

        k = min(limit * 15, int(self.clip_index.ntotal))
        best_scores: dict[int, float] = {}
        for variant in variants:
            query_embedding = self.clip_vec.encode_text(variant, normalize=True)
            scores, indices = self.clip_index.search(query_embedding, k)
            for rank, fid in enumerate(indices[0]):
                if fid == -1:
                    continue
                idx = int(self.clip_meta[int(fid)])
                score = float(scores[0][rank])
                current = best_scores.get(idx)
                if current is None or score > current:
                    best_scores[idx] = score

        ranked_candidates = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked_candidates:
            return []
        if ranked_candidates[0][1] < self.config.min_clip_score:
            self.logger.info(
                "Rejecting CLIP results for query=%r: top_score=%.3f < %.3f",
                query,
                ranked_candidates[0][1],
                self.config.min_clip_score,
            )
            return []

        out: list[int] = []
        for idx, score in ranked_candidates:
            if score < self.config.min_clip_score:
                continue
            out.append(idx)
            if len(out) >= limit:
                break
        return out

    def rerank_clip_over_candidates(self, query: str, candidates: list[int], topk: int = 10) -> list[int]:
        if self.clip_index is None or self.clip_vec is None or not candidates:
            return candidates[:topk]

        query_text = (query or "").strip()
        if not query_text:
            return candidates[:topk]

        variants = build_query_variants(query_text)
        if not variants:
            return candidates[:topk]

        candidate_set = set(candidates)
        search_k = min(2000, int(self.clip_index.ntotal))
        best_scores: dict[int, float] = {}
        for variant in variants:
            query_embedding = self.clip_vec.encode_text(variant, normalize=True)
            scores, indices = self.clip_index.search(query_embedding, search_k)
            for rank, fid in enumerate(indices[0]):
                if fid == -1:
                    continue
                idx = int(self.clip_meta[int(fid)])
                if idx not in candidate_set:
                    continue
                score = float(scores[0][rank])
                current = best_scores.get(idx)
                if current is None or score > current:
                    best_scores[idx] = score

        if not best_scores:
            return candidates[:topk]

        top_score = max(best_scores.values())
        if top_score < self.config.min_clip_rerank_score:
            self.logger.info(
                "Skipping CLIP rerank for query=%r: top_candidate_score=%.3f < %.3f",
                query,
                top_score,
                self.config.min_clip_rerank_score,
            )
            return candidates[:topk]

        ranked = [idx for idx, _ in sorted(best_scores.items(), key=lambda item: item[1], reverse=True)]
        out = ranked[:topk]
        if len(out) < topk:
            for idx in candidates:
                if idx not in out:
                    out.append(idx)
                if len(out) >= topk:
                    break
        return out

    def search(self, query: str, user_id: int | None = None, limit: int = 10) -> list[int]:
        lexical = self.search_keyword(query, limit=max(limit, 250))
        if lexical:
            results = self.rerank_clip_over_candidates(query, lexical, topk=limit)
        else:
            results = self.search_clip(query, limit=limit)
        return results
