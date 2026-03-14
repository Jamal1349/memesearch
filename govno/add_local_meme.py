import argparse
import os
import shutil
from datetime import datetime

from app_config import load_config
from storage import load_json_file, save_json_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a local meme overlay entry.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--description", required=True, help="Description used for search.")
    parser.add_argument("--alt", default="", help="Optional alt text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)

    os.makedirs(config.local_images_dir, exist_ok=True)
    local_items = load_json_file(config.local_memes_path, [])

    ext = os.path.splitext(args.image)[1] or ".jpg"
    filename = f"local_{len(local_items):05d}{ext}"
    target_path = os.path.join(config.local_images_dir, filename)
    shutil.copy2(args.image, target_path)

    local_items.append(
        {
            "image_path": target_path,
            "description": args.description,
            "alt": args.alt,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    )
    save_json_file(config.local_memes_path, local_items)
    print(f"Added local meme: {target_path}")


if __name__ == "__main__":
    main()
