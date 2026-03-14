import argparse
import glob
import json
import os
import re
from typing import Callable, Optional

from datasets import Dataset, load_dataset

from build_ocr_texts import extract_text, row_to_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "meme")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "meme_enriched")
DEFAULT_OCR_CACHE = os.path.join(BASE_DIR, "ocr_texts.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich a meme dataset with OCR plus LLM-generated retrieval fields."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Dataset root with parquet shards, usually ../meme.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the enriched dataset via save_to_disk().",
    )
    parser.add_argument(
        "--ocr-cache",
        default=DEFAULT_OCR_CACHE,
        help="JSON cache with OCR text by row index.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start row index.")
    parser.add_argument("--limit", type=int, default=0, help="How many rows to process. 0 means all rows.")
    parser.add_argument("--lang", default="rus+eng", help="OCR languages for Tesseract.")
    parser.add_argument("--save-every", type=int, default=200, help="Persist OCR cache every N processed rows.")
    parser.add_argument(
        "--caption-model",
        default="",
        help="Optional transformers image-to-text model used to add visual hints for the LLM prompt.",
    )
    parser.add_argument(
        "--llm-model",
        default="",
        help="Optional local instruct model used to generate intent_tags and query_expansions.",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=160,
        help="Maximum new tokens for LLM-based enrichment.",
    )
    parser.add_argument(
        "--keep-legacy-columns",
        action="store_true",
        help="Also populate legacy keywords/query_phrases columns from the new LLM fields.",
    )
    return parser.parse_args()


def load_local_parquet_dataset(input_dir: str) -> Dataset:
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "data", "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found under {input_dir}\\data")
    return load_dataset("parquet", data_files=parquet_files, split="train")


def load_json(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str, payload: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def text_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(text_value(item) for item in value if text_value(item)).strip()
    return str(value).strip()


def clean_text_for_prompt(text: str, max_chars: int = 700) -> str:
    text = text_value(text).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def normalize_piece(text: str) -> str:
    text = text_value(text).lower()
    text = text.replace(";", ",").replace("|", ",")
    text = re.sub(r"^[\s\-–—,.:]+|[\s\-–—,.:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_list_output(text: str, limit: int) -> list[str]:
    raw = text_value(text)
    if not raw:
        return []

    values: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[\n,;/]+", raw):
        value = normalize_piece(chunk)
        if not value:
            continue
        if ":" in value:
            prefix, rest = value.split(":", 1)
            if prefix in {"intent_tags", "query_expansions", "keywords", "queries", "tags"}:
                value = normalize_piece(rest)
        if len(value) < 3 or value.isdigit():
            continue
        if len(value.split()) > 8:
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
        if len(values) >= limit:
            break
    return values


def unique_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(text_value(item).split()).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def build_captioner(model_name: str) -> Optional[Callable]:
    if not model_name:
        return None

    from transformers import pipeline

    generator = pipeline("image-to-text", model=model_name)

    def captioner(image) -> str:
        result = generator(image)
        if not result:
            return ""
        return text_value(result[0].get("generated_text", ""))

    return captioner


def build_llm_enricher(args: argparse.Namespace) -> Optional[Callable[..., dict[str, list[str]]]]:
    if not args.llm_model:
        return None

    import torch
    from transformers import pipeline

    common_kwargs = {
        "model": args.llm_model,
        "device_map": "auto",
    }
    if torch.cuda.is_available():
        common_kwargs["torch_dtype"] = torch.float16

    try:
        generator = pipeline("text2text-generation", **common_kwargs)
        mode = "text2text"
    except Exception:
        generator = pipeline("text-generation", **common_kwargs)
        mode = "text-generation"

    def enrich_with_model(
        description: str,
        alt: str,
        ocr_text: str,
        caption: str,
        tag_limit: int = 12,
        query_limit: int = 8,
    ) -> dict[str, list[str]]:
        text_block = "\n".join(
            [
                f"description: {clean_text_for_prompt(description)}",
                f"alt: {clean_text_for_prompt(alt)}",
                f"ocr_text: {clean_text_for_prompt(ocr_text)}",
                f"caption: {clean_text_for_prompt(caption)}",
            ]
        ).strip()

        prompt = (
            "You are enriching a meme dataset for Russian search.\n"
            "Generate two compact retrieval fields.\n"
            "Field 1: intent_tags = short comma-separated tags about topic, tone, emotion, characters, and meme use-case.\n"
            "Field 2: query_expansions = short comma-separated user-like search queries in Russian.\n"
            "Rules:\n"
            "- Prefer Russian.\n"
            f"- Up to {tag_limit} intent_tags.\n"
            f"- Up to {query_limit} query_expansions.\n"
            "- Keep each item short.\n"
            "- No explanations.\n"
            "- No numbering.\n"
            "- No duplicates.\n\n"
            f"{text_block}\n\n"
            "intent_tags:\n"
            "query_expansions:"
        )

        if mode == "text2text":
            result = generator(prompt, max_new_tokens=args.llm_max_new_tokens, do_sample=False)
            output = result[0].get("generated_text", "")
        else:
            result = generator(
                prompt,
                max_new_tokens=args.llm_max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
            output = result[0].get("generated_text", "")

        intent_text = ""
        query_text = ""
        current_key = None
        for raw_line in text_value(output).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("intent_tags:"):
                current_key = "intent_tags"
                intent_text = line.split(":", 1)[1].strip()
                continue
            if lower.startswith("query_expansions:"):
                current_key = "query_expansions"
                query_text = line.split(":", 1)[1].strip()
                continue
            if current_key == "intent_tags":
                intent_text = (intent_text + ", " + line).strip(", ")
            elif current_key == "query_expansions":
                query_text = (query_text + ", " + line).strip(", ")

        return {
            "intent_tags": parse_list_output(intent_text, limit=tag_limit),
            "query_expansions": parse_list_output(query_text, limit=query_limit),
        }

    return enrich_with_model


def enrich_row(
    row: dict,
    idx: int,
    ocr_cache: dict[str, str],
    args: argparse.Namespace,
    captioner,
    llm_enricher,
) -> dict[str, str]:
    description = text_value(row.get("description"))
    alt = text_value(row.get("alt"))

    cache_key = str(idx)
    ocr_text = text_value(row.get("ocr_text")) or text_value(ocr_cache.get(cache_key))
    if not ocr_text:
        try:
            ocr_text = extract_text(row_to_image(row), args.lang)
        except Exception:
            ocr_text = ""
        ocr_cache[cache_key] = ocr_text

    caption = ""
    if captioner is not None:
        try:
            caption = text_value(captioner(row_to_image(row)))
        except Exception:
            caption = ""

    intent_tags: list[str] = []
    query_expansions: list[str] = []
    if llm_enricher is not None:
        try:
            enriched = llm_enricher(description, alt, ocr_text, caption, tag_limit=12, query_limit=8)
            intent_tags = unique_keep_order(enriched.get("intent_tags", []))[:12]
            query_expansions = unique_keep_order(enriched.get("query_expansions", []))[:8]
        except Exception:
            intent_tags = []
            query_expansions = []

    if not query_expansions:
        seeds = [
            clean_text_for_prompt(description, max_chars=80),
            clean_text_for_prompt(ocr_text, max_chars=80),
            clean_text_for_prompt(alt, max_chars=80),
        ]
        query_expansions = unique_keep_order([seed for seed in seeds if len(seed.split()) >= 2])[:4]

    return {
        "ocr_text": ocr_text,
        "intent_tags": ", ".join(intent_tags),
        "query_expansions": " || ".join(query_expansions),
        "keywords": ", ".join(intent_tags) if args.keep_legacy_columns else "",
        "query_phrases": " || ".join(query_expansions) if args.keep_legacy_columns else "",
    }


def main() -> None:
    args = parse_args()
    dataset = load_local_parquet_dataset(args.input_dir)
    total = len(dataset)
    end = total if args.limit <= 0 else min(total, args.start + args.limit)
    ocr_cache = load_json(args.ocr_cache)
    captioner = build_captioner(args.caption_model)
    llm_enricher = build_llm_enricher(args)

    ocr_values: list[str] = []
    intent_values: list[str] = []
    expansion_values: list[str] = []
    keyword_values: list[str] = []
    phrase_values: list[str] = []

    for idx in range(total):
        row = dataset[idx]
        if idx < args.start or idx >= end:
            ocr_values.append(text_value(row.get("ocr_text")))
            intent_values.append(text_value(row.get("intent_tags")))
            expansion_values.append(text_value(row.get("query_expansions")))
            keyword_values.append(text_value(row.get("keywords")))
            phrase_values.append(text_value(row.get("query_phrases")))
            continue

        enriched = enrich_row(row, idx, ocr_cache, args, captioner, llm_enricher)
        ocr_values.append(enriched["ocr_text"])
        intent_values.append(enriched["intent_tags"])
        expansion_values.append(enriched["query_expansions"])
        keyword_values.append(enriched["keywords"])
        phrase_values.append(enriched["query_phrases"])

        processed = idx - args.start + 1
        if processed % args.save_every == 0:
            save_json(args.ocr_cache, ocr_cache)
            print(f"processed={processed} rows, latest_idx={idx}")

    save_json(args.ocr_cache, ocr_cache)

    for column_name, values in (
        ("ocr_text", ocr_values),
        ("intent_tags", intent_values),
        ("query_expansions", expansion_values),
        ("keywords", keyword_values),
        ("query_phrases", phrase_values),
    ):
        if column_name in dataset.column_names:
            dataset = dataset.remove_columns(column_name)
        dataset = dataset.add_column(column_name, values)

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    print(f"saved enriched dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
