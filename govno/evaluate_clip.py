import argparse
import json
import os
from typing import Iterable, List

import faiss
import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm

from CLIP import STClipVectorizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "clip.index")
META_PATH = os.path.join(BASE_DIR, "clip_meta.npy")
INFO_PATH = os.path.join(BASE_DIR, "clip_index_info.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP text-to-image retrieval on the meme dataset."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="How many dataset rows to evaluate. Use 0 or a negative value for all rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for text encoding.",
    )
    parser.add_argument(
        "--query-field",
        choices=["description", "alt", "both"],
        default="both",
        help="Which dataset text field to use as the retrieval query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Cutoffs for Recall@K.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="HF model id or local checkpoint directory. Falls back to CLIP_MODEL_PATH/default.",
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
    return parser.parse_args()


def load_eval_dataset(args: argparse.Namespace):
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected DatasetDict at {args.dataset_path}")
        if args.dataset_split not in dataset:
            raise KeyError(f"Split '{args.dataset_split}' not found in {args.dataset_path}")
        return dataset[args.dataset_split], args.dataset_split

    return load_dataset(args.dataset_name, split=args.source_split), args.source_split


def load_queries(ds, query_field: str) -> List[tuple[int, str]]:
    queries: List[tuple[int, str]] = []

    for idx in range(len(ds)):
        row = ds[idx]
        target_id = int(row.get("original_id", idx))
        description = (row.get("description") or "").strip()
        alt = (row.get("alt") or "").strip()

        texts: List[str]
        if query_field == "description":
            texts = [description]
        elif query_field == "alt":
            texts = [alt]
        else:
            texts = [description, alt]

        for text in texts:
            if text:
                queries.append((target_id, text))

    return queries


def sample_queries(
    queries: List[tuple[int, str]], sample_size: int, seed: int
) -> List[tuple[int, str]]:
    if sample_size <= 0 or sample_size >= len(queries):
        return queries

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(queries), size=sample_size, replace=False)
    return [queries[int(i)] for i in indices]


def batched(items: List[tuple[int, str]], batch_size: int) -> Iterable[List[tuple[int, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def compute_rank(retrieved_ids: np.ndarray, target_id: int) -> int | None:
    matches = np.where(retrieved_ids == target_id)[0]
    if matches.size == 0:
        return None
    return int(matches[0]) + 1


def compute_ndcg_at_k(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def evaluate(
    vec: STClipVectorizer,
    index,
    meta: np.ndarray,
    queries: List[tuple[int, str]],
    batch_size: int,
    ks: List[int],
) -> dict:
    max_k = max(ks)
    search_k = min(max(int(index.ntotal), max_k), int(index.ntotal))

    ranks: List[int] = []
    ndcg_at_10: List[float] = []
    missing = 0

    total_batches = (len(queries) + batch_size - 1) // batch_size
    for batch in tqdm(
        batched(queries, batch_size),
        total=total_batches,
        desc="Evaluating",
    ):
        texts = [text for _, text in batch]
        embeddings = vec.encode_text(texts, normalize=True)
        _, found = index.search(embeddings, search_k)

        for row_idx, (target_id, _) in enumerate(batch):
            retrieved = np.array(
                [int(meta[int(fid)]) for fid in found[row_idx] if fid != -1],
                dtype=np.int64,
            )
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


def main() -> None:
    args = parse_args()

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")
    
    print("loading dataset...")
    ds, split_name = load_eval_dataset(args)

    print("building queries...")
    queries = load_queries(ds, args.query_field)

    queries = sample_queries(queries, args.sample_size, args.seed)

    if not queries:
        raise RuntimeError("No non-empty evaluation queries were found in the dataset.")

    print("loading index...")
    index = faiss.read_index(INDEX_PATH)
    meta = np.load(META_PATH)
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, "r", encoding="utf-8") as fh:
            index_info = json.load(fh)
        print(
            f"index_split: {index_info.get('dataset_split')} "
            f"from {index_info.get('dataset_source')}"
        )
    
    print("loading model...")
    vec = STClipVectorizer(model_name=args.model_name) if args.model_name else STClipVectorizer()
    print("evaluating...")
    results = evaluate(vec, index, meta, queries, args.batch_size, sorted(set(args.ks)))

    print("CLIP retrieval evaluation")
    print(f"dataset_split: {split_name}")
    print(f"query_field: {args.query_field}")
    print(f"evaluated_queries: {results['total_queries']}")
    print(f"missing_queries: {results['missing_queries']}")
    print(f"MRR: {results['mrr']:.4f}")
    print(f"NDCG@10: {results['ndcg@10']:.4f}")
    print(f"MeanRank: {results['mean_rank']:.2f}")
    print(f"MedianRank: {results['median_rank']:.2f}")
    for k, value in results["recall"].items():
        print(f"Recall@{k}: {value:.4f}")


if __name__ == "__main__":
    main()
