import argparse

from app_config import configure_logging, load_config
from search_engine import SearchEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test merged meme search locally.")
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument("--limit", type=int, default=10, help="How many results to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    config = load_config()
    search_engine = SearchEngine(config, logger)

    results = search_engine.search(args.query, user_id=0, limit=args.limit)
    if not results:
        print("No results")
        return

    for idx in results:
        row = search_engine.row(idx)
        source = row.get("source", "hf")
        title = row.get("description") or row.get("alt") or f"meme #{idx}"
        print(f"{idx}\t[{source}]\t{title}")


if __name__ == "__main__":
    main()
