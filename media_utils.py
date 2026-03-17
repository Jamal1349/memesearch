import base64
import os

from aiogram.types import BufferedInputFile

from app_config import resolve_runtime_path

PLACEHOLDER_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def get_image_bytes(row) -> bytes:
    if isinstance(row, dict) and row.get("image_b64"):
        return base64.b64decode(row["image_b64"])
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.getenv("APP_DATA_DIR", base_dir)
    image_data = row["image"]
    if isinstance(image_data, dict) and "bytes" in image_data:
        image_bytes = image_data["bytes"]
        if isinstance(image_bytes, (bytes, bytearray)):
            return bytes(image_bytes)
    if isinstance(image_data, dict) and "path" in image_data:
        image_path = image_data["path"]
        if isinstance(image_path, str):
            resolved_path = resolve_runtime_path(image_path, data_dir, base_dir)
            if os.path.exists(resolved_path):
                with open(resolved_path, "rb") as f:
                    return f.read()
        if isinstance(image_path, str) and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return f.read()
    raise ValueError(f"Unknown image format: {type(image_data)}")


def create_input_file(row) -> BufferedInputFile:
    try:
        image_bytes = get_image_bytes(row)
        idx = row.get("idx") if isinstance(row, dict) else row.name
        return BufferedInputFile(image_bytes, filename=f"meme_{idx}.jpg")
    except Exception:
        return BufferedInputFile(PLACEHOLDER_1PX, filename="error.jpg")
