import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    VisionTextDualEncoderModel,
)


DEFAULT_CLIP_BASE_MODEL = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_FINETUNED_DIRS = [
    os.path.join(BASE_DIR, "clip_finetuned"),
    os.path.join(os.path.dirname(BASE_DIR), "clip_finetuned"),
]
DEFAULT_CLIP_MODEL = os.getenv("CLIP_MODEL_PATH") or next(
    (path for path in DEFAULT_LOCAL_FINETUNED_DIRS if os.path.exists(path)),
    os.getenv("CLIP_BASE_MODEL", DEFAULT_CLIP_BASE_MODEL),
)
DEFAULT_OPENCLIP_ARCH = os.getenv("OPENCLIP_ARCH", "xlm-roberta-base-ViT-B-32")
DEFAULT_OPENCLIP_PRETRAINED = os.getenv("OPENCLIP_PRETRAINED", "laion5b_s13b_b90k")
OPENCLIP_CHECKPOINT_EXTENSIONS = (".pt", ".pth")


class STClipVectorizer:
    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.openclip_mode = self._is_openclip_checkpoint(model_name)
        self.openclip_preprocess = None
        self.openclip_tokenizer = None

        if self.openclip_mode:
            self._load_openclip_checkpoint(model_name)
            return

        self.config = AutoConfig.from_pretrained(model_name)
        self.model_type = getattr(self.config, "model_type", "")

        if self.model_type == "vision-text-dual-encoder":
            self.model = VisionTextDualEncoderModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = self._load_dual_tokenizer(model_name)
            self.image_processor = self._load_dual_image_processor(model_name)
            self.processor = None
            self.max_text_length = self._resolve_text_max_length(self.tokenizer)
        else:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.tokenizer = None
            self.image_processor = None
            self.max_text_length = None

        self.model.eval()

    def encode_text(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
            if self.openclip_mode:
                tokens = self.openclip_tokenizer(texts).to(self.device)
                embeddings = self.model.encode_text(tokens)
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                return embeddings.cpu().numpy().astype("float32")
            if self.model_type == "vision-text-dual-encoder":
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            embeddings = self._get_text_embeddings(inputs)
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.cpu().numpy().astype("float32")

    def encode_image(self, images, normalize: bool = True) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        pil_images = [self._to_pil(image) for image in images]
        with torch.no_grad():
            if self.openclip_mode:
                pixels = torch.stack([self.openclip_preprocess(image) for image in pil_images]).to(self.device)
                embeddings = self.model.encode_image(pixels)
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                return embeddings.cpu().numpy().astype("float32")
            if self.model_type == "vision-text-dual-encoder":
                inputs = self.image_processor(images=pil_images, return_tensors="pt")
            else:
                inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            embeddings = self._get_image_embeddings(inputs)
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.cpu().numpy().astype("float32")

    def _get_text_embeddings(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model.get_text_features(**inputs)
        return self._coerce_embedding_tensor(outputs, getattr(self.model, "text_projection", None))

    def _get_image_embeddings(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model.get_image_features(**inputs)
        return self._coerce_embedding_tensor(outputs, getattr(self.model, "visual_projection", None))

    def _coerce_embedding_tensor(self, outputs, projection) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs

        for attr_name in ("text_embeds", "image_embeds", "embeds"):
            value = getattr(outputs, attr_name, None)
            if isinstance(value, torch.Tensor):
                return value

        pooled = getattr(outputs, "pooler_output", None)
        if isinstance(pooled, torch.Tensor):
            return self._maybe_project(pooled, projection)

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if isinstance(last_hidden_state, torch.Tensor):
            cls_state = last_hidden_state[:, 0]
            return self._maybe_project(cls_state, projection)

        raise TypeError(f"Unsupported embedding output type: {type(outputs)}")

    def _maybe_project(self, embeddings: torch.Tensor, projection) -> torch.Tensor:
        if projection is None:
            return embeddings

        in_features = getattr(projection, "in_features", None)
        out_features = getattr(projection, "out_features", None)
        current_dim = embeddings.shape[-1]

        if current_dim == in_features:
            return projection(embeddings)
        if current_dim == out_features:
            return embeddings
        return embeddings

    def _load_dual_tokenizer(self, model_name: str):
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception:
            text_source = getattr(self.config, "text_config", None)
            text_name = getattr(text_source, "_name_or_path", None) or getattr(
                self.config, "text_config_name", None
            )
            if not text_name:
                raise
            return AutoTokenizer.from_pretrained(text_name)

    def _load_dual_image_processor(self, model_name: str):
        try:
            return AutoImageProcessor.from_pretrained(model_name)
        except Exception:
            vision_source = getattr(self.config, "vision_config", None)
            vision_name = getattr(vision_source, "_name_or_path", None) or getattr(
                self.config, "vision_config_name", None
            )

            if vision_name:
                try:
                    return AutoImageProcessor.from_pretrained(vision_name)
                except Exception:
                    pass

            image_size = 384
            vision_config = getattr(self.config, "vision_config", None)
            if vision_config is not None:
                image_size = getattr(vision_config, "image_size", image_size)

            return CLIPImageProcessor(
                size={"shortest_edge": image_size},
                crop_size={"height": image_size, "width": image_size},
                resample=3,
                do_resize=True,
                do_center_crop=True,
                do_convert_rgb=True,
                do_rescale=True,
                rescale_factor=1 / 255,
                do_normalize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
            )

    def _resolve_text_max_length(self, tokenizer) -> int:
        max_len = getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 100000:
            return 512
        return min(max_len, 512)

    def _to_pil(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict):
            if image.get("path"):
                return Image.open(image["path"]).convert("RGB")
            if image.get("bytes"):
                from io import BytesIO

                return Image.open(BytesIO(image["bytes"])).convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _is_openclip_checkpoint(self, model_name: str) -> bool:
        if not model_name:
            return False
        lower_path = str(model_name).lower()
        return os.path.isfile(model_name) and lower_path.endswith(OPENCLIP_CHECKPOINT_EXTENSIONS)

    def _load_openclip_checkpoint(self, checkpoint_path: str) -> None:
        try:
            import open_clip
        except Exception as exc:
            raise RuntimeError(
                "open_clip_torch is required to load .pt CLIP checkpoints. "
                "Install it with: pip install open_clip_torch"
            ) from exc

        arch = os.getenv("OPENCLIP_ARCH", DEFAULT_OPENCLIP_ARCH)
        pretrained = os.getenv("OPENCLIP_PRETRAINED", DEFAULT_OPENCLIP_PRETRAINED)
        try:
            self.model, _, self.openclip_preprocess = open_clip.create_model_and_transforms(
                arch,
                pretrained=None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize OpenCLIP architecture '{arch}' for local checkpoint '{checkpoint_path}'. "
                "Set OPENCLIP_ARCH to the architecture used during training."
            ) from exc
        self.openclip_tokenizer = open_clip.get_tokenizer(arch)
        self.model = self.model.to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                key = key.replace("module.", "")
                cleaned_state_dict[key] = value
            state_dict = cleaned_state_dict

        incompatible = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = getattr(incompatible, "missing_keys", [])
        unexpected_keys = getattr(incompatible, "unexpected_keys", [])
        if missing_keys and unexpected_keys:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_path}' is incompatible with OPENCLIP_ARCH='{arch}'. "
                f"Missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}. "
                f"If this checkpoint expects pretrained base weights, set OPENCLIP_PRETRAINED='{pretrained}' "
                "and ensure those weights are available locally."
            )
        self.model_type = "open_clip"
        self.model.eval()
