import json
import os
import tempfile
from typing import Any

from app_config import AppConfig


def load_json_file(path: str, default: Any = None) -> Any:
    if default is None:
        default = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json_file(path: str, data: Any) -> None:
    dir_name = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_name, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    os.replace(temp_path, path)


class BotStorage:
    def __init__(self, config: AppConfig):
        self.config = config
        self.blocked_users_path = os.path.join(config.data_dir, "blocked_users.json")
        self.favorites = load_json_file(config.favorites_path, {})
        self.file_id_cache = load_json_file(config.file_ids_path, {})
        self.blocked_users = set(load_json_file(self.blocked_users_path, []))

    def get_favorites(self, user_id: int | str) -> list[int]:
        return list(self.favorites.get(str(user_id), []))

    def add_favorite(self, user_id: int | str, meme_idx: int) -> bool:
        key = str(user_id)
        favs = set(self.favorites.get(key, []))
        if meme_idx in favs:
            return False
        favs.add(meme_idx)
        self.favorites[key] = list(favs)
        self.save_favorites()
        return True

    def remove_favorite(self, user_id: int | str, meme_idx: int) -> bool:
        key = str(user_id)
        favs = list(self.favorites.get(key, []))
        if meme_idx not in favs:
            return False
        favs.remove(meme_idx)
        self.favorites[key] = favs
        self.save_favorites()
        return True

    def save_favorites(self) -> None:
        save_json_file(self.config.favorites_path, self.favorites)

    def get_file_id(self, meme_idx: int) -> str | None:
        return self.file_id_cache.get(str(meme_idx))

    def set_file_id(self, meme_idx: int, file_id: str) -> None:
        self.file_id_cache[str(meme_idx)] = file_id
        save_json_file(self.config.file_ids_path, self.file_id_cache)

    def delete_file_id(self, meme_idx: int) -> None:
        if str(meme_idx) in self.file_id_cache:
            self.file_id_cache.pop(str(meme_idx), None)
            save_json_file(self.config.file_ids_path, self.file_id_cache)

    def load_warmup_state(self) -> dict:
        return load_json_file(self.config.warmup_state_path, {"next_idx": 0, "ok": 0, "fail": 0})

    def save_warmup_state(self, state: dict) -> None:
        save_json_file(self.config.warmup_state_path, state)

    def append_local_meme(
        self,
        image_path: str,
        description: str,
        alt: str = "",
        query_phrases: str = "",
        keywords: str = "",
        ocr_text: str = "",
    ) -> dict:
        items = load_json_file(self.config.local_memes_path, [])
        record = {
            "image_path": image_path,
            "description": description,
            "alt": alt,
            "query_phrases": query_phrases,
            "keywords": keywords,
            "ocr_text": ocr_text,
        }
        items.append(record)
        save_json_file(self.config.local_memes_path, items)
        return record

    def delete_local_meme(self, image_path: str) -> bool:
        items = load_json_file(self.config.local_memes_path, [])
        updated_items = [item for item in items if item.get("image_path") != image_path]
        if len(updated_items) == len(items):
            return False
        save_json_file(self.config.local_memes_path, updated_items)
        return True

    def purge_meme_references(self, meme_idx: int) -> None:
        changed = False
        for user_id, favorites in list(self.favorites.items()):
            filtered = [fav for fav in favorites if fav != meme_idx]
            if len(filtered) != len(favorites):
                self.favorites[user_id] = filtered
                changed = True
        if changed:
            self.save_favorites()
        self.delete_file_id(meme_idx)

    def save_blocked_users(self) -> None:
        save_json_file(self.blocked_users_path, sorted(self.blocked_users))

    def is_blocked(self, user_id: int | str) -> bool:
        return str(user_id) in self.blocked_users

    def block_user(self, user_id: int | str) -> bool:
        key = str(user_id)
        if key in self.blocked_users:
            return False
        self.blocked_users.add(key)
        self.save_blocked_users()
        return True

    def unblock_user(self, user_id: int | str) -> bool:
        key = str(user_id)
        if key not in self.blocked_users:
            return False
        self.blocked_users.remove(key)
        self.save_blocked_users()
        return True

    def list_blocked_users(self) -> list[str]:
        return sorted(self.blocked_users)
