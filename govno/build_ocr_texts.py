import argparse
import io
import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional

from datasets import load_dataset
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "ocr_texts.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OCR text for meme images.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path to output JSON.")
    parser.add_argument("--start", type=int, default=0, help="Start dataset index.")
    parser.add_argument("--limit", type=int, default=0, help="How many rows to process. 0 means all.")
    parser.add_argument("--lang", default="rus+eng", help="Tesseract language pack(s).")
    parser.add_argument("--save-every", type=int, default=100, help="Save progress every N rows.")
    return parser.parse_args()


def row_to_image(row) -> Image.Image:
    image_data = row["image"]
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image_data)}")


def ocr_with_pytesseract(image: Image.Image, lang: str) -> Optional[str]:
    try:
        import pytesseract  # type: ignore
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Vadim\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    except Exception:
        return None
    return pytesseract.image_to_string(image, lang=lang).strip()


def ocr_with_tesseract_cli(image: Image.Image, lang: str) -> Optional[str]:
    if not shutil.which("tesseract"):
        return None

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, "image.png")
        output_base = os.path.join(temp_dir, "ocr")
        image.save(image_path)
        proc = subprocess.run(
            ["tesseract", image_path, output_base, "-l", lang],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        txt_path = output_base + ".txt"
        if not os.path.exists(txt_path):
            return None
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()


def extract_text(image: Image.Image, lang: str) -> str:
    text = ocr_with_pytesseract(image, lang)
    if text is not None:
        return text

    text = ocr_with_tesseract_cli(image, lang)
    if text is not None:
        return text

    raise RuntimeError("Neither pytesseract nor the tesseract CLI is available.")


def load_existing(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_output(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    ds = load_dataset("DIvest1ng/meme", split="train")
    total = len(ds)
    end = total if args.limit <= 0 else min(total, args.start + args.limit)
    output = load_existing(args.output)

    for idx in range(args.start, end):
        key = str(idx)
        if key in output:
            continue
        image = row_to_image(ds[idx])
        output[key] = extract_text(image, args.lang)

        if (idx - args.start + 1) % args.save_every == 0:
            save_output(args.output, output)
            print(f"saved: {idx + 1}/{end}")

    save_output(args.output, output)
    print(f"done: {len(output)} rows in {args.output}")


if __name__ == "__main__":
    main()
