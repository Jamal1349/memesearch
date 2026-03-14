import argparse
import io
import math
import os
from dataclasses import dataclass
from bisect import bisect_right

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    VisionTextDualEncoderModel,
)


DEFAULT_CLIP_BASE_MODEL = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
DEFAULT_MODEL = os.getenv("CLIP_BASE_MODEL") or os.getenv("CLIP_MODEL_PATH", DEFAULT_CLIP_BASE_MODEL)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_MODE_CHOICES = [
    "description",
    "alt",
    "both",
    "query_phrases",
    "keywords",
    "ocr_text",
    "enriched",
    "hybrid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on meme image-text pairs.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base CLIP checkpoint.")
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
    parser.add_argument(
        "--local-memes-json",
        default=os.path.join(BASE_DIR, "local_memes.json"),
        help="Optional local meme records to append to training split.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Training split name for disk datasets.",
    )
    parser.add_argument(
        "--validation-split",
        default="validation",
        help="Validation split name for disk datasets.",
    )
    parser.add_argument("--text-mode", choices=TEXT_MODE_CHOICES, default="enriched")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(BASE_DIR, "clip_finetuned"),
        help="Where to save the best checkpoint.",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def coerce_text_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(coerce_text_value(item) for item in value if coerce_text_value(item)).strip()
    return str(value).strip()


def choose_text(row: dict, text_mode: str) -> str:
    description = coerce_text_value(row.get("description"))
    alt = coerce_text_value(row.get("alt"))
    query_phrases = coerce_text_value(row.get("query_phrases"))
    keywords = coerce_text_value(row.get("keywords"))
    ocr_text = coerce_text_value(row.get("ocr_text"))

    if text_mode == "description":
        return description
    if text_mode == "alt":
        return alt
    if text_mode == "both":
        return " ".join(part for part in [description, alt] if part).strip()
    if text_mode == "query_phrases":
        return query_phrases
    if text_mode == "keywords":
        return keywords
    if text_mode == "ocr_text":
        return ocr_text
    if text_mode == "enriched":
        return " ".join(part for part in [query_phrases, keywords, ocr_text] if part).strip()
    if text_mode == "hybrid":
        return " ".join(
            part for part in [query_phrases, keywords, ocr_text, description, alt] if part
        ).strip()
    raise ValueError(f"Unsupported text mode: {text_mode}")


def row_has_text(row: dict, text_mode: str) -> bool:
    return bool(choose_text(row, text_mode))


def load_local_memes(local_memes_json: str) -> Dataset | None:
    if not local_memes_json or not os.path.exists(local_memes_json):
        return None
    ds = load_dataset("json", data_files=local_memes_json, split="train")
    return ds if len(ds) else None


def load_training_splits(args: argparse.Namespace) -> tuple[list[Dataset], Dataset | None]:
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected DatasetDict at {args.dataset_path}")
        train_ds = dataset[args.train_split]
        val_ds = dataset[args.validation_split] if args.validation_split in dataset else None
    elif args.dataset_name:
        train_ds = load_dataset(args.dataset_name, split=args.source_split)
        val_ds = None
    else:
        raise FileNotFoundError(
            f"Dataset path not found: {args.dataset_path}. Pass --dataset-name as fallback."
        )

    train_parts = [train_ds.filter(lambda row: row_has_text(row, args.text_mode))]
    local_ds = load_local_memes(args.local_memes_json)
    if local_ds is not None:
        train_parts.append(local_ds.filter(lambda row: row_has_text(row, args.text_mode)))
    if val_ds is not None:
        val_ds = val_ds.filter(lambda row: row_has_text(row, args.text_mode))

    return train_parts, val_ds


def row_to_pil(row: dict) -> Image.Image:
    if "image_path" in row and row["image_path"]:
        return Image.open(row["image_path"]).convert("RGB")

    image = row.get("image")
    if isinstance(image, dict):
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
        if image.get("bytes"):
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise ValueError(f"Unsupported image payload: {type(image)}")


class MemeClipDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[Dataset], text_mode: str):
        self.datasets = datasets
        self.text_mode = text_mode
        self.offsets: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.offsets.append(total)

    def __len__(self) -> int:
        return self.offsets[-1] if self.offsets else 0

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        dataset_idx = bisect_right(self.offsets, idx)
        start = 0 if dataset_idx == 0 else self.offsets[dataset_idx - 1]
        row = self.datasets[dataset_idx][int(idx - start)]
        return row_to_pil(row), choose_text(row, self.text_mode)


def resolve_text_max_length(processor_or_tokenizer, explicit_max_length: int) -> int:
    if explicit_max_length and explicit_max_length > 0:
        return explicit_max_length
    max_len = getattr(processor_or_tokenizer, "model_max_length", None)
    if not isinstance(max_len, int) or max_len <= 0 or max_len > 100000:
        return 128
    return min(max_len, 512)


def load_text_tokenizer(model_name: str, config) -> object:
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        text_source = getattr(config, "text_config", None)
        text_name = getattr(text_source, "_name_or_path", None) or getattr(config, "text_config_name", None)
        if not text_name:
            raise
        return AutoTokenizer.from_pretrained(text_name)


def load_image_processor(model_name: str, config) -> object:
    try:
        return AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        vision_source = getattr(config, "vision_config", None)
        vision_name = getattr(vision_source, "_name_or_path", None) or getattr(
            config, "vision_config_name", None
        )
        if not vision_name:
            raise
        return AutoImageProcessor.from_pretrained(vision_name)


@dataclass
class Collator:
    text_tokenizer: object
    image_processor: object
    max_text_length: int

    def __call__(self, batch: list[tuple[Image.Image, str]]) -> dict[str, torch.Tensor]:
        images = [image for image, _ in batch]
        texts = [text for _, text in batch]
        text_inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )
        image_inputs = self.image_processor(images=images, return_tensors="pt")
        return {
            **text_inputs,
            **image_inputs,
        }


def load_model_bundle(model_name: str, device: str) -> tuple[torch.nn.Module, str, object, object]:
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "")
    if model_type == "vision-text-dual-encoder":
        model = VisionTextDualEncoderModel.from_pretrained(model_name).to(device)
        text_tokenizer = load_text_tokenizer(model_name, config)
        image_processor = load_image_processor(model_name, config)
    else:
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        text_tokenizer = processor.tokenizer
        image_processor = processor.image_processor
    return model, model_type, text_tokenizer, image_processor


def build_dataloader(
    datasets: list[Dataset],
    text_tokenizer: object,
    image_processor: object,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    torch_dataset = MemeClipDataset(datasets, args.text_mode)
    max_text_length = resolve_text_max_length(text_tokenizer, args.max_text_length)
    return DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=Collator(
            text_tokenizer=text_tokenizer,
            image_processor=image_processor,
            max_text_length=max_text_length,
        ),
    )


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: str) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch, device)
            outputs = model(**batch, return_loss=True)
            losses.append(float(outputs.loss.detach().cpu().item()))
    return sum(losses) / len(losses) if losses else math.nan


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model_name}")
    model, _, text_tokenizer, image_processor = load_model_bundle(args.model_name, device)

    print("Loading datasets...")
    train_parts, val_ds = load_training_splits(args)
    train_rows = sum(len(part) for part in train_parts)
    print(f"Train rows: {train_rows}")
    if val_ds is not None:
        print(f"Validation rows: {len(val_ds)}")

    train_loader = build_dataloader(train_parts, text_tokenizer, image_processor, args, shuffle=True)
    val_loader = (
        build_dataloader([val_ds], text_tokenizer, image_processor, args, shuffle=False)
        if val_ds is not None
        else None
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1 if warmup_steps > 0 else 1.0,
        total_iters=max(warmup_steps, 1),
    )

    best_val = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if step <= warmup_steps:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().cpu().item())
            if step % 10 == 0 or step == len(train_loader):
                avg_loss = running_loss / step
                print(f"epoch={epoch} step={step}/{len(train_loader)} train_loss={avg_loss:.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        print(f"epoch={epoch} train_loss={train_loss:.4f}")

        if val_loader is None:
            model.save_pretrained(args.output_dir)
            text_tokenizer.save_pretrained(args.output_dir)
            image_processor.save_pretrained(args.output_dir)
            continue

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(args.output_dir)
            text_tokenizer.save_pretrained(args.output_dir)
            image_processor.save_pretrained(args.output_dir)
            print(f"Saved new best checkpoint to {args.output_dir}")

    print("Training finished.")


if __name__ == "__main__":
    main()
