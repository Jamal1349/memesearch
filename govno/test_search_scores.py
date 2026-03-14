import argparse

from app_config import configure_logging, load_config
from search_engine import (
    SearchEngine,
    build_query_variants,
    coerce_text_value,
    normalize_query_key,
    normalize_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect BM25 and CLIP scores for one or more search queries."
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to inspect. Pass multiple times for multiple queries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many top results to print per search method.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read queries from stdin until an empty line is entered.",
    )
    return parser.parse_args()


def format_title(search_engine: SearchEngine, idx: int) -> str:
    title = search_engine.title_for_idx(idx).replace("\n", " ").strip()
    return title[:120] if len(title) > 120 else title


def format_field(value: object, limit: int = 140) -> str:
    text = coerce_text_value(value).replace("\n", " ").strip()
    if not text:
        return "-"
    return text[:limit] if len(text) > limit else text


def print_row_details(search_engine: SearchEngine, idx: int) -> None:
    row = search_engine.row(idx)
    print(f"    description: {format_field(row.get('description'))}")
    print(f"    alt:         {format_field(row.get('alt'))}")
    print(f"    query:       {format_field(row.get('query_phrases'))}")
    print(f"    keywords:    {format_field(row.get('keywords'))}")
    print(f"    ocr:         {format_field(row.get('ocr_text'))}")


def inspect_bm25(search_engine: SearchEngine, query: str, limit: int) -> list[tuple[int, float]]:
    if search_engine.bm25_index is None:
        return []

    query_key = normalize_query_key(query)[:200]
    tokens = normalize_tokens(query_key, max_tokens=12)
    if not tokens:
        return []
    return search_engine.bm25_index.score(tokens, topn=limit)


def inspect_clip(search_engine: SearchEngine, query: str, limit: int) -> list[tuple[int, float, str]]:
    if search_engine.clip_index is None or search_engine.clip_vec is None:
        return []

    variants = build_query_variants(query)
    if not variants:
        return []

    k = min(max(limit * 5, limit), int(search_engine.clip_index.ntotal))
    best_scores: dict[int, tuple[float, str]] = {}
    for variant in variants:
        embedding = search_engine.clip_vec.encode_text(variant, normalize=True)
        scores, indices = search_engine.clip_index.search(embedding, k)
        for rank, fid in enumerate(indices[0]):
            if fid == -1:
                continue
            idx = int(search_engine.clip_meta[int(fid)])
            score = float(scores[0][rank])
            current = best_scores.get(idx)
            if current is None or score > current[0]:
                best_scores[idx] = (score, variant)

    ranked = sorted(best_scores.items(), key=lambda item: item[1][0], reverse=True)
    return [(idx, score, variant) for idx, (score, variant) in ranked[:limit]]


def print_bm25(search_engine: SearchEngine, query: str, limit: int) -> None:
    results = inspect_bm25(search_engine, query, limit)
    tokens = normalize_tokens(normalize_query_key(query)[:200], max_tokens=12)
    print("\nBM25")
    print(f"tokens: {tokens}")
    if not results:
        print("no results")
        return

    for rank, (idx, score) in enumerate(results, start=1):
        print(f"{rank:>2}. idx={idx:<6} score={score:>8.4f} title={format_title(search_engine, idx)}")
        print_row_details(search_engine, idx)


def print_clip(search_engine: SearchEngine, query: str, limit: int) -> None:
    results = inspect_clip(search_engine, query, limit)
    variants = build_query_variants(query)
    print("\nCLIP")
    print(f"variants: {variants}")
    if not results:
        print("no results")
        return

    for rank, (idx, score, variant) in enumerate(results, start=1):
        print(
            f"{rank:>2}. idx={idx:<6} score={score:>8.4f} variant={variant!r} "
            f"title={format_title(search_engine, idx)}"
        )
        print_row_details(search_engine, idx)


def print_final_decision(search_engine: SearchEngine, query: str, limit: int) -> None:
    print("\nFINAL")
    print(
        f"thresholds: min_bm25={search_engine.config.min_bm25_score:.4f}, "
        f"min_clip={search_engine.config.min_clip_score:.4f}, "
        f"min_clip_rerank={search_engine.config.min_clip_rerank_score:.4f}"
    )

    bm25_raw = inspect_bm25(search_engine, query, max(limit * 5, 20))
    if bm25_raw:
        top_bm25 = bm25_raw[0][1]
        bm25_pass = top_bm25 >= search_engine.config.min_bm25_score
        print(f"bm25_top_score: {top_bm25:.4f} -> {'PASS' if bm25_pass else 'REJECT'}")
    else:
        bm25_pass = False
        print("bm25_top_score: none -> REJECT")

    clip_raw = inspect_clip(search_engine, query, max(limit * 5, 20))
    if clip_raw:
        top_clip = clip_raw[0][1]
        clip_pass = top_clip >= search_engine.config.min_clip_score
        print(f"clip_top_score: {top_clip:.4f} -> {'PASS' if clip_pass else 'REJECT'}")
    else:
        clip_pass = False
        print("clip_top_score: none -> REJECT")

    lexical_candidates = search_engine.search_keyword(query, user_id=-1, limit=250)
    print(f"bm25_candidates_after_threshold: {len(lexical_candidates)}")
    if lexical_candidates:
        reranked = search_engine.rerank_clip_over_candidates(query, lexical_candidates, topk=limit)
        print("reranked_bm25_by_clip:")
        for rank, idx in enumerate(reranked, start=1):
            print(f"{rank:>2}. idx={idx:<6} title={format_title(search_engine, idx)}")
            print_row_details(search_engine, idx)
    else:
        print("reranked_bm25_by_clip: -")

    final_results = search_engine.search(query, user_id=-1, limit=limit)
    if not final_results:
        print("final_search_results: none")
        return

    print("final_search_results:")
    for rank, idx in enumerate(final_results, start=1):
        print(f"{rank:>2}. idx={idx:<6} title={format_title(search_engine, idx)}")
        print_row_details(search_engine, idx)


def run_query(search_engine: SearchEngine, query: str, limit: int) -> None:
    query = (query or "").strip()
    if not query:
        return

    print("\n" + "=" * 80)
    print(f"query: {query}")
    print_bm25(search_engine, query, limit)
    print_clip(search_engine, query, limit)
    print_final_decision(search_engine, query, limit)


def collect_queries(args: argparse.Namespace) -> list[str]:
    queries = list(args.queries or [])
    if args.interactive:
        print("Enter queries one per line. Submit an empty line to stop.")
        while True:
            query = input("query> ").strip()
            if not query:
                break
            queries.append(query)
    return queries


def main() -> None:
    args = parse_args()
    queries = collect_queries(args)
    if not queries:
        raise SystemExit("Pass --query or use --interactive.")

    logger = configure_logging()
    config = load_config()
    search_engine = SearchEngine(config, logger)

    for query in queries:
        run_query(search_engine, query, args.limit)


if __name__ == "__main__":
    main()
