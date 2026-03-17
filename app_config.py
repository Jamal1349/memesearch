import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


DEFAULT_CLIP_BASE_MODEL = "M-CLIP/XLM-Roberta-Large-Vit-B-32"


@dataclass(frozen=True)
class AppConfig:
    token: str
    cache_chat_id: Optional[int]
    base_dir: str
    data_dir: str
    dataset_path: str
    dataset_name: str
    dataset_split: str
    clip_model_path: str
    clip_index_path: str
    clip_meta_path: str
    file_ids_path: str
    favorites_path: str
    interaction_logs_path: str
    warmup_state_path: str
    local_memes_path: str
    local_images_dir: str
    min_bm25_score: float
    min_clip_score: float
    min_clip_rerank_score: float
    min_clip_fallback_score: float
    inline_limit: int = 50
    inline_cache_time: int = 30
    query_token_limit: int = 512
    query_history_max_queries: int = 20
    query_history_max_results: int = 50
    admin_user_ids: frozenset[int] = frozenset()


def configure_logging() -> logging.Logger:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.getenv("APP_DATA_DIR", base_dir)
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(data_dir, "logs.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return logging.getLogger("memebot")


def load_config() -> AppConfig:
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN не задан")

    cache_chat_id_env = os.getenv("CACHE_CHAT_ID")
    cache_chat_id = int(cache_chat_id_env) if cache_chat_id_env else None

    admin_user_ids = frozenset()
    admin_env = os.getenv("ADMIN_USER_IDS", "").strip()
    if admin_env:
        admin_user_ids = frozenset(int(x) for x in admin_env.split(",") if x.strip())

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.getenv("APP_DATA_DIR", base_dir)
    os.makedirs(data_dir, exist_ok=True)
    default_dataset_path = os.path.join(base_dir, "meme_enriched")
    if not os.path.exists(default_dataset_path):
        default_dataset_path = os.path.join(base_dir, "dataset_splits")
    dataset_path = os.getenv("DATASET_PATH", default_dataset_path)
    dataset_name = os.getenv("DATASET_NAME", "DIvest1ng/meme")
    dataset_split = os.getenv("DATASET_SPLIT", "train")
    min_bm25_score = float(os.getenv("MIN_BM25_SCORE", "4.0"))
    min_clip_score = float(os.getenv("MIN_CLIP_SCORE", "0.23"))
    min_clip_rerank_score = float(os.getenv("MIN_CLIP_RERANK_SCORE", "0.35"))
    min_clip_fallback_score = float(os.getenv("MIN_CLIP_FALLBACK_SCORE", "0.28"))
    clip_model_path = os.getenv("CLIP_MODEL_PATH")
    if not clip_model_path:
        finetuned_candidates = [
            os.path.join(base_dir, "clip_finetuned"),
            os.path.join(os.path.dirname(base_dir), "clip_finetuned"),
        ]
        finetuned_dir = next((path for path in finetuned_candidates if os.path.exists(path)), None)
        clip_model_path = finetuned_dir or os.getenv("CLIP_BASE_MODEL", DEFAULT_CLIP_BASE_MODEL)
    clip_index_path = os.getenv("CLIP_INDEX_PATH", os.path.join(base_dir, "clip.index"))
    clip_meta_path = os.getenv("CLIP_META_PATH", os.path.join(base_dir, "clip_meta.npy"))

    return AppConfig(
        token=token,
        cache_chat_id=cache_chat_id,
        base_dir=base_dir,
        data_dir=data_dir,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        min_bm25_score=min_bm25_score,
        min_clip_score=min_clip_score,
        min_clip_rerank_score=min_clip_rerank_score,
        min_clip_fallback_score=min_clip_fallback_score,
        clip_model_path=clip_model_path,
        clip_index_path=clip_index_path,
        clip_meta_path=clip_meta_path,
        file_ids_path=os.path.join(data_dir, "file_ids.json"),
        favorites_path=os.path.join(data_dir, "favorites.json"),
        interaction_logs_path=os.path.join(data_dir, "interaction_logs.jsonl"),
        warmup_state_path=os.path.join(data_dir, "warmup_state.json"),
        local_memes_path=os.path.join(data_dir, "local_memes.json"),
        local_images_dir=os.path.join(data_dir, "local_memes"),
        admin_user_ids=admin_user_ids,
    )
