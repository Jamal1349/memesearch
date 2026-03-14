import argparse
import inspect
import json
import os
import random
import re
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.getenv("E5_MODEL_NAME", "intfloat/multilingual-e5-base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an E5-style text retriever on the meme dataset."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base embedding checkpoint.")
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(BASE_DIR, "dataset_splits"),
        help="Path created by split_dataset.py via save_to_disk().",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Fallback HF dataset name if --dataset-path does not exist.",
    )
    parser.add_argument("--source-split", default="train", help="HF split used with --dataset-name.")
    parser.add_argument("--train-split", default="train", help="Training split name for disk datasets.")
    parser.add_argument(
        "--validation-split",
        default="validation",
        help="Optional validation split name for disk datasets.",
    )
    parser.add_argument(
        "--local-memes-json",
        default=os.path.join(BASE_DIR, "local_memes.json"),
        help="Optional local meme records to append to training split.",
    )
    parser.add_argument(
        "--query-source",
        choices=["mixed", "query_expansions", "query_phrases", "description", "alt", "ocr_text"],
        default="mixed",
        help="Which fields to use as positive training queries.",
    )
    parser.add_argument(
        "--max-queries-per-doc",
        type=int,
        default=4,
        help="Maximum positive queries generated per meme.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=1200,
        help="Truncate passage text to this many characters.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=0, help="Override total training steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(BASE_DIR, "e5_finetuned"),
        help="Where to save the trained model.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        parts = [part.strip() for part in text.split("||")]
    else:
        parts = [part.strip() for part in text.split(",")]

    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = normalize_space(part)
        if len(part) < 2:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(part)
    return cleaned


def short_text(text: str, max_words: int) -> str:
    words = normalize_space(text).split()
    return " ".join(words[:max_words]).strip()


def build_document_text(row: dict, max_chars: int) -> str:
    query_text = coerce_text_value(row.get("query_expansions")) or coerce_text_value(row.get("query_phrases"))
    keywords_text = coerce_text_value(row.get("intent_tags")) or coerce_text_value(row.get("keywords"))
    ocr_text = coerce_text_value(row.get("ocr_text"))
    description = coerce_text_value(row.get("description"))
    alt = coerce_text_value(row.get("alt"))

    parts = [
        query_text,
        keywords_text,
        ocr_text,
        description,
        alt,
    ]
    text = normalize_space(" ".join(part for part in parts if part))
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


def load_local_memes(local_memes_json: str) -> Dataset | None:
    if not local_memes_json or not os.path.exists(local_memes_json):
        return None
    dataset = load_dataset("json", data_files=local_memes_json, split="train")
    return dataset if len(dataset) else None


@dataclass
class LoadedSplits:
    train_parts: list[Dataset]
    validation: Dataset | None


def load_training_splits(args: argparse.Namespace) -> LoadedSplits:
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
        if isinstance(dataset, DatasetDict):
            train_ds = dataset[args.train_split]
            validation_ds = dataset[args.validation_split] if args.validation_split in dataset else None
        else:
            train_ds = dataset
            validation_ds = None
    elif args.dataset_name:
        train_ds = load_dataset(args.dataset_name, split=args.source_split)
        validation_ds = None
    else:
        raise FileNotFoundError(
            f"Dataset path not found: {args.dataset_path}. Pass --dataset-name as fallback."
        )

    train_parts = [train_ds]
    local_ds = load_local_memes(args.local_memes_json)
    if local_ds is not None:
        train_parts.append(local_ds)

    return LoadedSplits(train_parts=train_parts, validation=validation_ds)


def rows_to_examples(dataset: Dataset, args: argparse.Namespace) -> list[InputExample]:
    examples: list[InputExample] = []

    for row in dataset:
        passage = build_document_text(row, args.max_doc_chars)
        if len(passage) < 8:
            continue

        queries = build_query_texts(row, args.query_source, args.max_queries_per_doc)
        for query in queries:
            examples.append(
                InputExample(
                    texts=[
                        f"query: {query}",
                        f"passage: {passage}",
                    ]
                )
            )

    return examples


def merge_train_parts(parts: list[Dataset]) -> Dataset:
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def save_training_args(args: argparse.Namespace, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    args_path = os.path.join(output_dir, "train_args.json")
    with open(args_path, "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("WANDB_DISABLED", "true")

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print("Loading dataset...")
    loaded = load_training_splits(args)
    train_ds = merge_train_parts(loaded.train_parts)
    print(f"Train rows: {len(train_ds)}")
    if loaded.validation is not None:
        print(f"Validation rows: {len(loaded.validation)}")

    print("Building training pairs...")
    train_examples = rows_to_examples(train_ds, args)
    if not train_examples:
        raise RuntimeError("No training examples were generated. Check your dataset fields.")
    print(f"Train pairs: {len(train_examples)}")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    save_training_args(args, args.output_dir)

    print("Training...")
    fit_kwargs = {
        "train_objectives": [(train_dataloader, train_loss)],
        "epochs": args.epochs,
        "warmup_steps": warmup_steps,
        "optimizer_params": {"lr": args.learning_rate},
        "weight_decay": args.weight_decay,
        "output_path": args.output_dir,
        "save_best_model": False,
        "checkpoint_path": os.path.join(args.output_dir, "checkpoints"),
        "checkpoint_save_steps": max(len(train_dataloader), 1),
        "show_progress_bar": True,
    }
    if args.max_steps > 0:
        fit_kwargs["max_steps"] = args.max_steps

    # sentence-transformers fit() kwargs vary across versions; pass only supported args.
    fit_signature = inspect.signature(model.fit)
    supported_kwargs = {
        key: value for key, value in fit_kwargs.items() if key in fit_signature.parameters
    }
    model.fit(**supported_kwargs)

    # Some sentence-transformers versions focus on checkpoints/trainer artifacts.
    # Force a final SentenceTransformer export in the requested output directory.
    model.save(args.output_dir)

    modules_path = os.path.join(args.output_dir, "modules.json")
    if not os.path.exists(modules_path):
        raise RuntimeError(
            f"Training finished but {modules_path} was not created. "
            "Model export failed; cannot run evaluate_e5.py with this output."
        )
    print(f"Saved model to: {args.output_dir}")


if __name__ == "__main__":
    main()
