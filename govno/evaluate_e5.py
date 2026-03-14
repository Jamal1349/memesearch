import argparse
import json
import os
import re
from typing import Iterable

import faiss
import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.getenv("E5_MODEL_PATH") or os.getenv("E5_MODEL_NAME", "intfloat/multilingual-e5-base")
INDEX_PATH = os.path.join(BASE_DIR, "e5.index")
META_PATH = os.path.join(BASE_DIR, "e5_meta.npy")
INFO_PATH = os.path.join(BASE_DIR, "e5_index_info.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate E5 text-to-text retrieval on the meme dataset."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="How many evaluation queries to score. Use 0 or a negative value for all queries.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for text encoding.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="HF model id or local checkpoint directory.",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(BASE_DIR, "dataset_splits"),
        help="Path created by split_dataset.py via save_to_disk().",
    )
    parser.add_argument(
        "--dataset-name",
        default="DIvest1ng/meme",
        help="Fallback HF dataset name if --dataset-path does not exist.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["train", "validation", "test"],
        default="test",
        help="Which split to evaluate when loading from --dataset-path.",
    )
    parser.add_argument(
        "--source-split",
        default="train",
        help="HF split used with --dataset-name fallback.",
    )
    parser.add_argument(
        "--query-source",
        choices=["mixed", "query_expansions", "query_phrases", "description", "alt", "ocr_text"],
        default="mixed",
        help="Which fields to use as evaluation queries.",
    )
    parser.add_argument(
        "--max-queries-per-doc",
        type=int,
        default=4,
        help="Maximum number of evaluation queries to use per meme.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=1200,
        help="Truncate indexed document text to this many characters.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Cutoffs for Recall@K.",
    )
    parser.add_argument(
        "--save-index",
        action="store_true",
        help="Persist the FAISS index and metadata after evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def coerce_text_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple)):
        parts = [coerce_text_value(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def split_query_chunks(text: str) -> list[str]:
    text = coerce_text_value(text)
    if not text:
        return []

    if "||" in text:
        raw_parts = text.split("||")
    else:
        raw_parts = text.split(",")

    out: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        part = normalize_space(part)
        if len(part) < 2:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out


def short_text(text: str, max_words: int) -> str:
    words = normalize_space(text).split()
    return " ".join(words[:max_words]).strip()


def build_document_text(row: dict, max_chars: int) -> str:
    query_text = coerce_text_value(row.get("query_expansions")) or coerce_text_value(row.get("query_phrases"))
    keywords_text = coerce_text_value(row.get("intent_tags")) or coerce_text_value(row.get("keywords"))
    ocr_text = coerce_text_value(row.get("ocr_text"))
    description = coerce_text_value(row.get("description"))
    alt = coerce_text_value(row.get("alt"))

    text = normalize_space(" ".join(part for part in [query_text, keywords_text, ocr_text, description, alt] if part))
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def build_query_texts(row: dict, source: str, max_queries_per_doc: int) -> list[str]:
    query_expansions = split_query_chunks(row.get("query_expansions"))
    query_phrases = split_query_chunks(row.get("query_phrases"))
    description = short_text(coerce_text_value(row.get("description")), max_words=12)
    alt = short_text(coerce_text_value(row.get("alt")), max_words=12)
    ocr_text = short_text(coerce_text_value(row.get("ocr_text")), max_words=16)

    if source == "query_expansions":
        candidates = query_expansions
    elif source == "query_phrases":
        candidates = query_phrases
    elif source == "description":
        candidates = [description]
    elif source == "alt":
        candidates = [alt]
    elif source == "ocr_text":
        candidates = [ocr_text]
    else:
        candidates = query_expansions + query_phrases + [description, alt, ocr_text]

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = normalize_space(candidate)
        if len(candidate) < 2:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= max_queries_per_doc:
            break
    return out


def load_eval_dataset(args: argparse.Namespace):
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected DatasetDict at {args.dataset_path}")
        if args.dataset_split not in dataset:
            raise KeyError(f"Split '{args.dataset_split}' not found in {args.dataset_path}")
        return dataset[args.dataset_split], args.dataset_split

    return load_dataset(args.dataset_name, split=args.source_split), args.source_split


def build_corpus(ds, max_doc_chars: int) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    ids: list[int] = []

    for idx in range(len(ds)):
        row = ds[idx]
        target_id = int(row.get("original_id", idx))
        passage = build_document_text(row, max_doc_chars)
        if len(passage) < 8:
            continue
        ids.append(target_id)
        texts.append(f"passage: {passage}")

    return texts, np.array(ids, dtype=np.int64)


def build_queries(ds, query_source: str, max_queries_per_doc: int) -> list[tuple[int, str]]:
    queries: list[tuple[int, str]] = []

    for idx in range(len(ds)):
        row = ds[idx]
        target_id = int(row.get("original_id", idx))
        for query in build_query_texts(row, query_source, max_queries_per_doc):
            queries.append((target_id, f"query: {query}"))

    return queries


def sample_queries(
    queries: list[tuple[int, str]], sample_size: int, seed: int
) -> list[tuple[int, str]]:
    if sample_size <= 0 or sample_size >= len(queries):
        return queries

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(queries), size=sample_size, replace=False)
    return [queries[int(i)] for i in indices]


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch in tqdm(batched(texts, batch_size), total=total_batches, desc="Encoding"):
        embeddings = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        chunks.append(embeddings.astype("float32"))
    return np.vstack(chunks)


def compute_rank(retrieved_ids: np.ndarray, target_id: int) -> int | None:
    matches = np.where(retrieved_ids == target_id)[0]
    if matches.size == 0:
        return None
    return int(matches[0]) + 1


def compute_ndcg_at_k(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def evaluate(index, meta: np.ndarray, query_embeddings: np.ndarray, queries: list[tuple[int, str]], ks: list[int]) -> dict:
    max_k = max(ks)
    search_k = min(max(int(index.ntotal), max_k), int(index.ntotal))
    scores, found = index.search(query_embeddings, search_k)

    ranks: list[int] = []
    ndcg_at_10: list[float] = []
    missing = 0

    for row_idx, (target_id, _) in enumerate(queries):
        _ = scores[row_idx]
        retrieved = np.array([int(meta[int(fid)]) for fid in found[row_idx] if fid != -1], dtype=np.int64)
        rank = compute_rank(retrieved, target_id)
        if rank is None:
            missing += 1
            rank = search_k + 1
        ranks.append(rank)
        ndcg_at_10.append(compute_ndcg_at_k(rank, 10))

    total = len(ranks)
    recall = {k: sum(rank <= k for rank in ranks) / total for k in ks}
    mrr = sum(1.0 / rank for rank in ranks) / total

    return {
        "total_queries": total,
        "missing_queries": missing,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mrr": mrr,
        "ndcg@10": float(np.mean(ndcg_at_10)),
        "recall": recall,
    }


def save_index(index, meta: np.ndarray, info: dict) -> None:
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, meta)
    with open(INFO_PATH, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    print("loading dataset...")
    ds, split_name = load_eval_dataset(args)

    print("building corpus...")
    corpus_texts, corpus_ids = build_corpus(ds, args.max_doc_chars)
    if len(corpus_texts) == 0:
        raise RuntimeError("No corpus documents were generated. Check dataset text fields.")

    print("building queries...")
    queries = build_queries(ds, args.query_source, args.max_queries_per_doc)
    queries = sample_queries(queries, args.sample_size, args.seed)
    if not queries:
        raise RuntimeError("No evaluation queries were generated. Check query-source and dataset fields.")

    print("loading model...")
    model = SentenceTransformer(args.model_name)

    print("encoding corpus...")
    corpus_embeddings = encode_texts(model, corpus_texts, args.batch_size)

    print("building FAISS index...")
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    print("encoding queries...")
    query_texts = [query_text for _, query_text in queries]
    query_embeddings = encode_texts(model, query_texts, args.batch_size)

    print("evaluating...")
    results = evaluate(index, corpus_ids, query_embeddings, queries, sorted(set(args.ks)))

    print("E5 retrieval evaluation")
    print(f"dataset_split: {split_name}")
    print(f"query_source: {args.query_source}")
    print(f"evaluated_queries: {results['total_queries']}")
    print(f"missing_queries: {results['missing_queries']}")
    print(f"MRR: {results['mrr']:.4f}")
    print(f"NDCG@10: {results['ndcg@10']:.4f}")
    print(f"MeanRank: {results['mean_rank']:.2f}")
    print(f"MedianRank: {results['median_rank']:.2f}")
    for k, value in results["recall"].items():
        print(f"Recall@{k}: {value:.4f}")

    if args.save_index:
        save_index(
            index,
            corpus_ids,
            {
                "model_name": args.model_name,
                "dataset_split": split_name,
                "dataset_source": args.dataset_path if os.path.exists(args.dataset_path) else args.dataset_name,
                "total_documents": int(index.ntotal),
                "embedding_dim": int(corpus_embeddings.shape[1]),
                "query_source": args.query_source,
                "max_doc_chars": args.max_doc_chars,
            },
        )
        print(f"saved_index: {INDEX_PATH}")
        print(f"saved_meta: {META_PATH}")
        print(f"saved_info: {INFO_PATH}")


if __name__ == "__main__":
    main()
