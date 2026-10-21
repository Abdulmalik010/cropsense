"""Image classifier: fine-tuned MobileNetV3 via timm."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.config import Settings, class_by_id, load_label_catalog, load_settings

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


class ClassifierError(Exception):
    """Raised when an image cannot be classified."""


def _load_torch_modules() -> tuple[Any, Any, Any]:
    import timm
    import torch
    from torchvision import transforms

    return torch, timm, transforms


@lru_cache(maxsize=1)
def _build_transform() -> Any:
    _, _, transforms = _load_torch_modules()
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def image_from_bytes(data: bytes) -> Image.Image:
    """Decode image bytes into an RGB PIL image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ClassifierError("Could not read the image file.") from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class CropClassifier:
    """Loads `models/classifier.pt` and returns top-k class scores."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.catalog = load_label_catalog(self.settings.labels_path)
        self.class_names = [item["id"] for item in self.catalog["classes"]]
        self.model_name = self.catalog.get("model_name", "mobilenetv3_small_100")
        self._model: Any = None
        self._torch: Any = None

    def weights_exist(self) -> bool:
        """Return True if the checkpoint file is present."""
        return self.settings.classifier_weights.is_file()

    def load(self) -> None:
        """Load weights into memory. Safe to call more than once."""
        if self._model is not None:
            return
        if not self.weights_exist():
            raise ClassifierError(
                f"Missing classifier weights at {self.settings.classifier_weights}"
            )
        torch, timm, _ = _load_torch_modules()
        checkpoint = torch.load(
            self.settings.classifier_weights, map_location="cpu", weights_only=False
        )
        state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        saved_classes = (
            checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
        )
        if saved_classes and list(saved_classes) != self.class_names:
            raise ClassifierError(
                "Checkpoint class names do not match models/labels.json. Retrain or update labels."
            )
        model_name = (
            checkpoint.get("model_name", self.model_name)
            if isinstance(checkpoint, dict)
            else self.model_name
        )
        model = timm.create_model(
            model_name, pretrained=False, num_classes=len(self.class_names)
        )
        model.load_state_dict(state)
        model.eval()
        self._model = model
        self._torch = torch

    def predict(self, image: Image.Image, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return top-k predictions with display labels and confidence."""
        self.load()
        k = top_k or self.settings.top_k_predictions
        tensor = _build_transform()(image).unsqueeze(0)
        with self._torch.no_grad():
            logits = self._model(tensor)
            probabilities = self._torch.softmax(logits, dim=1)[0]
        k = min(k, len(self.class_names))
        scores, indices = probabilities.topk(k)
        results: list[dict[str, Any]] = []
        for score, index in zip(scores.tolist(), indices.tolist(), strict=True):
            class_id = self.class_names[index]
            meta = class_by_id(class_id, self.catalog)
            results.append(
                {
                    "id": class_id,
                    "label": meta["display"],
                    "crop": meta["crop"],
                    "kb_file": meta["kb_file"],
                    "confidence": round(float(score), 4),
                }
            )
        return results


def weights_status(weights_path: Path | None = None) -> dict[str, Any]:
    """Describe classifier availability for `/health`."""
    settings = load_settings()
    path = weights_path or settings.classifier_weights
    return {
        "ready": path.is_file(),
        "weights": str(path),
        "labels": str(settings.labels_path),
    }
