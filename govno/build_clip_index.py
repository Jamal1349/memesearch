import io
import time
import logging
import argparse
import json
import os
import numpy as np
import faiss
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

from CLIP import STClipVectorizer


INDEX_PATH = "clip.index"
META_PATH = "clip_meta.npy"
INFO_PATH = "clip_index_info.json"
LOG_FILE = "clip_build.log"

BATCH = 64
LOG_EVERY = 100  
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("clip_build")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a FAISS index with a CLIP checkpoint.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="HF model id or local checkpoint directory. Falls back to CLIP_MODEL_PATH/default.",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_splits"),
        help="Path created by split_dataset.py via save_to_disk().",
    )
    parser.add_argument(
        "--dataset-name",
        default="DIvest1ng/meme",
        help="Fallback HF dataset name if --dataset-path does not exist.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["all", "train", "validation", "test"],
        default="all",
        help="Which split to index when loading from --dataset-path.",
    )
    parser.add_argument(
        "--source-split",
        default="train",
        help="HF split used with --dataset-name fallback.",
    )
    return parser.parse_args()


def load_index_dataset(args):
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected DatasetDict at {args.dataset_path}")
        if args.dataset_split == "all":
            split_names = list(dataset.keys())
            combined = concatenate_datasets([dataset[split_name] for split_name in split_names])
            return combined, "all", args.dataset_path
        if args.dataset_split not in dataset:
            raise KeyError(f"Split '{args.dataset_split}' not found in {args.dataset_path}")
        return dataset[args.dataset_split], args.dataset_split, args.dataset_path

    return load_dataset(args.dataset_name, split=args.source_split), args.source_split, args.dataset_name


# ---------------- IMAGE ----------------
def row_to_pil(row) -> Image.Image:
    img = row["image"]
    if isinstance(img, dict) and img.get("path"):
        return Image.open(img["path"]).convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    raise ValueError(f"Unknown image type: {type(img)}")


# ---------------- MAIN ----------------
def main():
    args = parse_args()
    logger.info("🚀 Start building CLIP index")

    start_time = time.time()

    logger.info("Loading dataset...")
    ds, split_name, dataset_source = load_index_dataset(args)
    total = len(ds)
    logger.info("Dataset source: %s", dataset_source)
    logger.info("Dataset split: %s", split_name)
    logger.info(f"Dataset size: {total}")

    logger.info("Loading models...")
    vec = STClipVectorizer(model_name=args.model_name) if args.model_name else STClipVectorizer()

    embs = []
    meta = []

    batch_imgs = []
    batch_ids = []

    processed = 0
    errors = 0

    for i in tqdm(range(total), desc="Encoding images"):
        try:
            row = ds[i]
            batch_imgs.append(row_to_pil(row))
            batch_ids.append(int(row.get("original_id", i)))
        except Exception as e:
            errors += 1
            logger.warning(f"Image decode error idx={i}: {e}")
            continue

        if len(batch_imgs) == BATCH:
            emb = vec.encode_image(batch_imgs)
            embs.append(emb)
            meta.extend(batch_ids)

            processed += len(batch_imgs)
            batch_imgs = []
            batch_ids = []

            if processed % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / speed if speed > 0 else 0

                logger.info(
                    f"processed={processed}/{total} | "
                    f"errors={errors} | "
                    f"speed={speed:.2f} img/sec | "
                    f"ETA={eta/60:.1f} min"
                )

    if batch_imgs:
        emb = vec.encode_image(batch_imgs)
        embs.append(emb)
        meta.extend(batch_ids)
        processed += len(batch_imgs)

    logger.info("Stacking vectors...")
    embs = np.vstack(embs).astype("float32")
    meta = np.array(meta, dtype=np.int64)

    logger.info("Building FAISS index...")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    logger.info("Saving index...")
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, meta)
    with open(INFO_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model_name": args.model_name,
                "dataset_source": dataset_source,
                "dataset_split": split_name,
                "total": int(total),
                "embedding_dim": int(embs.shape[1]),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    total_time = time.time() - start_time
    logger.info("✅ DONE")
    logger.info(f"Total processed: {processed}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Time: {total_time/60:.1f} min")
    logger.info(f"Speed avg: {processed/total_time:.2f} img/sec")


if __name__ == "__main__":
    main()
