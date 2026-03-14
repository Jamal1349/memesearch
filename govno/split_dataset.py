import argparse
import os

from datasets import DatasetDict, load_dataset, load_from_disk


DEFAULT_DATASET = "DIvest1ng/meme"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test splits for CLIP retraining."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name or local dataset path.",
    )
    parser.add_argument(
        "--source-split",
        default="train",
        help="Source split to split further.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples for train.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples for test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before split.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_splits"),
        help="Where to save the split dataset with save_to_disk().",
    )
    parser.add_argument(
        "--text-fields",
        nargs="+",
        default=["query_phrases", "keywords", "ocr_text", "description", "alt"],
        help="Text fields that can make a row valid for training.",
    )
    parser.add_argument(
        "--drop-empty-text",
        action="store_true",
        help="Drop rows where all text fields are empty.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r <= 0 for r in ratios):
        raise ValueError("All split ratios must be positive.")

    total = sum(ratios)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")


def has_any_text(row: dict, text_fields: list[str]) -> bool:
    for field in text_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def load_source_dataset(dataset_name_or_path: str, source_split: str):
    if os.path.exists(dataset_name_or_path):
        dataset = load_from_disk(dataset_name_or_path)
        if isinstance(dataset, DatasetDict):
            if source_split not in dataset:
                raise KeyError(f"Split '{source_split}' not found in {dataset_name_or_path}")
            return dataset[source_split]
        return dataset
    return load_dataset(dataset_name_or_path, split=source_split)


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    print(f"Loading dataset: {args.dataset} [{args.source_split}]")
    ds = load_source_dataset(args.dataset, args.source_split)
    ds = ds.map(lambda _, idx: {"original_id": idx}, with_indices=True)
    print(f"Loaded rows: {len(ds)}")

    if args.drop_empty_text:
        before = len(ds)
        ds = ds.filter(lambda row: has_any_text(row, args.text_fields))
        print(f"Rows after empty-text filter: {len(ds)} (dropped {before - len(ds)})")

    first_split = ds.train_test_split(
        test_size=(args.val_ratio + args.test_ratio),
        seed=args.seed,
        shuffle=True,
    )

    holdout = first_split["test"]
    val_share_of_holdout = args.val_ratio / (args.val_ratio + args.test_ratio)
    second_split = holdout.train_test_split(
        test_size=(1.0 - val_share_of_holdout),
        seed=args.seed,
        shuffle=True,
    )

    split_ds = DatasetDict(
        {
            "train": first_split["train"],
            "validation": second_split["train"],
            "test": second_split["test"],
        }
    )

    os.makedirs(args.output_dir, exist_ok=True)
    split_ds.save_to_disk(args.output_dir)

    print("Saved splits:")
    for split_name, split in split_ds.items():
        print(f"  {split_name}: {len(split)} rows")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
