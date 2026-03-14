import base64
import os

from aiogram.types import BufferedInputFile


PLACEHOLDER_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def get_image_bytes(row) -> bytes:
    image_data = row["image"]
    if isinstance(image_data, dict) and "bytes" in image_data:
        image_bytes = image_data["bytes"]
        if isinstance(image_bytes, (bytes, bytearray)):
            return bytes(image_bytes)
    if isinstance(image_data, dict) and "path" in image_data:
        image_path = image_data["path"]
        if isinstance(image_path, str) and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return f.read()
    raise ValueError(f"Unknown image format: {type(image_data)}")


def create_input_file(row) -> BufferedInputFile:
    try:
        image_bytes = get_image_bytes(row)
        return BufferedInputFile(image_bytes, filename=f"meme_{row.name}.jpg")
    except Exception:
        return BufferedInputFile(PLACEHOLDER_1PX, filename="error.jpg")
