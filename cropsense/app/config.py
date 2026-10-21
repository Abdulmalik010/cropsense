"""Load CropSense settings from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, str(default)).lower()
    return raw in {"1", "true", "yes", "on"}


SUPPORTED_LANGUAGES = {
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "pcm": "Nigerian Pidgin",
}

DISCLAIMER_EN = (
    "This is decision support for farmers, not a replacement for a trained "
    "agronomist or agricultural extension officer."
)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the Flask app and ML pipeline."""

    debug: bool
    host: str
    port: int
    ollama_host: str
    llm_model: str
    embed_model: str
    confidence_threshold: float
    top_k_predictions: int
    rag_n_results: int
    classifier_weights: Path
    labels_path: Path
    chroma_path: Path
    knowledge_dir: Path
    max_image_bytes: int


def load_settings() -> Settings:
    """Return settings from `.env` and process environment."""
    return Settings(
        debug=_env_bool("FLASK_DEBUG", True),
        host=_env("HOST", "0.0.0.0"),
        port=_env_int("PORT", 5000),
        ollama_host=_env("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        llm_model=_env("LLM_MODEL", "gemma4:e4b"),
        embed_model=_env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", 0.60),
        top_k_predictions=_env_int("TOP_K_PREDICTIONS", 3),
        rag_n_results=_env_int("RAG_N_RESULTS", 5),
        classifier_weights=_resolve(_env("CLASSIFIER_WEIGHTS", "models/classifier.pt")),
        labels_path=_resolve(_env("LABELS_PATH", "models/labels.json")),
        chroma_path=_resolve(_env("CHROMA_PATH", "data/chroma")),
        knowledge_dir=_resolve(_env("KNOWLEDGE_DIR", "app/knowledge")),
        max_image_bytes=_env_int("MAX_IMAGE_BYTES", 8 * 1024 * 1024),
    )


def _resolve(relative: str) -> Path:
    path = Path(relative)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_label_catalog(labels_path: Path | None = None) -> dict[str, Any]:
    """Load class metadata from `models/labels.json`."""
    path = labels_path or load_settings().labels_path
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "classes" not in data:
        raise ValueError(f"Invalid labels file (missing 'classes'): {path}")
    return data


def class_ids(catalog: dict[str, Any] | None = None) -> list[str]:
    """Return ordered classifier class ids."""
    data = catalog if catalog is not None else load_label_catalog()
    return [item["id"] for item in data["classes"]]


def class_by_id(class_id: str, catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    """Look up display name, crop, and knowledge-base file for a class id."""
    data = catalog if catalog is not None else load_label_catalog()
    for item in data["classes"]:
        if item["id"] == class_id:
            return item
    raise KeyError(f"Unknown class id: {class_id}")
