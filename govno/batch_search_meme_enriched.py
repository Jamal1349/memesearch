import argparse
import os

from app_config import AppConfig, configure_logging, load_config
from search_engine import SearchEngine, coerce_text_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run many search queries against meme_enriched and print rich result fields."
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Explicit query to run. Pass multiple times.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="How many queries to run when auto-generating them from the dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many search results to print per query.",
    )

    parser.add_argument(
        "--query-source",
        choices=["mixed", "query_phrases", "keywords", "description", "alt"],
        default="mixed",
        help="Which field to use when auto-generating queries.",
    )
    parser.add_argument(
        "--show-ocr",
        action="store_true",
        help="Also print OCR text for each result.",
    )
    return parser.parse_args()


def build_offline_config() -> AppConfig:
    try:
        return load_config()
    except RuntimeError:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_dir, "meme_enriched")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(base_dir, "dataset_splits")

        return AppConfig(
            token="offline",
            cache_chat_id=None,
            base_dir=base_dir,
            dataset_path=dataset_path,
            dataset_name="DIvest1ng/meme",
            dataset_split="train",
            clip_model_path=os.getenv("CLIP_MODEL_PATH", "M-CLIP/XLM-Roberta-Large-Vit-B-32"),
            clip_index_path=os.path.join(base_dir, "clip.index"),
            clip_meta_path=os.path.join(base_dir, "clip_meta.npy"),
            file_ids_path=os.path.join(base_dir, "file_ids.json"),
            favorites_path=os.path.join(base_dir, "favorites.json"),
            warmup_state_path=os.path.join(base_dir, "warmup_state.json"),
            local_memes_path=os.path.join(base_dir, "local_memes.json"),
            local_images_dir=os.path.join(base_dir, "local_memes"),
            min_bm25_score=float(os.getenv("MIN_BM25_SCORE", "4.0")),
            min_clip_score=float(os.getenv("MIN_CLIP_SCORE", "0.23")),
            min_clip_rerank_score=float(os.getenv("MIN_CLIP_RERANK_SCORE", "0.35")),
        )


def format_field(value: object, limit: int = 180) -> str:
    text = coerce_text_value(value).replace("\n", " ").strip()
    if not text:
        return "-"
    return text[:limit] if len(text) <= limit else text[: limit - 3] + "..."


def short_query_phrase(text: str) -> str:
    parts = [part.strip() for part in text.split("||") if part.strip()]
    if not parts:
        return ""
    best = min(parts, key=len)
    return best[:120].strip()


def short_keywords(text: str) -> str:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return ""
    return " ".join(parts[:6])[:120].strip()


def short_text(text: str, max_words: int = 8) -> str:
    words = [word for word in text.replace("\n", " ").split() if word.strip()]
    if not words:
        return ""
    return " ".join(words[:max_words]).strip()


def row_to_candidate_queries(row, source: str) -> list[str]:
    description = coerce_text_value(row.get("description"))
    alt = coerce_text_value(row.get("alt"))
    query_phrases = coerce_text_value(row.get("query_phrases"))
    keywords = coerce_text_value(row.get("keywords"))

    if source == "query_phrases":
        return [short_query_phrase(query_phrases)]
    if source == "keywords":
        return [short_keywords(keywords)]
    if source == "description":
        return [short_text(description)]
    if source == "alt":
        return [short_text(alt)]

    return [
        short_query_phrase(query_phrases),
        short_keywords(keywords),
        short_text(description),
        short_text(alt),
    ]


def collect_auto_queries(search_engine: SearchEngine, count: int, source: str) -> list[str]:
    seen: set[str] = set()
    queries: list[str] = []

    if search_engine.df is None:
        return queries

    for _, row in search_engine.df.iterrows():
        for query in row_to_candidate_queries(row, source):
            query = (query or "").strip()
            if len(query) < 2:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(query)
            if len(queries) >= count:
                return queries
    return queries


def print_result(search_engine: SearchEngine, idx: int, rank: int, show_ocr: bool) -> None:
    row = search_engine.row(idx)
    print(f"  {rank}. idx={idx} source={format_field(row.get('source'), 20)}")
    print(f"     description:   {format_field(row.get('description'))}")
    print(f"     alt:           {format_field(row.get('alt'))}")
    print(f"     keywords:      {format_field(row.get('keywords'))}")
    print(f"     query_phrases: {format_field(row.get('query_phrases'))}")
    if show_ocr:
        print(f"     ocr_text:      {format_field(row.get('ocr_text'))}")


def run_query(search_engine: SearchEngine, query: str, limit: int, show_ocr: bool) -> None:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    results = search_engine.search(query, user_id=-1, limit=limit)
    if not results:
        print("  no results")
        return

    for rank, idx in enumerate(results, start=1):
        print_result(search_engine, idx, rank, show_ocr=show_ocr)


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    config = build_offline_config()
    search_engine = SearchEngine(config, logger)

    queries = list(args.queries or [])
    if not queries:
        queries = collect_auto_queries(search_engine, args.count, args.query_source)

    if not queries:
        raise SystemExit("No queries were provided or generated.")

    print(f"dataset_path: {config.dataset_path}")
    print(f"queries_to_run: {len(queries)}")
    print(f"query_source: {args.query_source}")
    print(f"results_per_query: {args.limit}")

    for query in queries:
        run_query(search_engine, query, args.limit, show_ocr=args.show_ocr)


if __name__ == "__main__":
    main()



