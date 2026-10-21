"""HTTP routes for CropSense."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, current_app, jsonify, render_template, request

from app.advisor import chroma_status, generate_advice, ollama_alive, with_disclaimer
from app.classifier import ClassifierError, CropClassifier, image_from_bytes, weights_status
from app.config import SUPPORTED_LANGUAGES, Settings
from app.messages import BAD_IMAGE, LOW_CONFIDENCE, MISSING_MODEL, disclaimer, normalize_language, text

bp = Blueprint("cropsense", __name__)


def _settings() -> Settings:
    return current_app.config["CROPSENSE_SETTINGS"]


def _classifier() -> CropClassifier:
    stored = current_app.config.get("CROPSENSE_CLASSIFIER")
    if stored is None:
        stored = CropClassifier(_settings())
        current_app.config["CROPSENSE_CLASSIFIER"] = stored
    return stored


@bp.get("/")
def index() -> str:
    """Serve the mobile-first single page."""
    return render_template("index.html", languages=SUPPORTED_LANGUAGES)


@bp.get("/health")
def health():
    """Return classifier and Ollama status."""
    settings = _settings()
    ollama = ollama_alive(settings)
    classifier = weights_status(settings.classifier_weights)
    kb = chroma_status(settings)
    ok = classifier["ready"] and ollama.get("ready") is True
    return jsonify(
        {
            "status": "ok" if ok else "degraded",
            "classifier": classifier,
            "ollama": ollama,
            "knowledge_base": kb,
            "llm_model": settings.llm_model,
            "confidence_threshold": settings.confidence_threshold,
        }
    )


@bp.post("/api/diagnose")
def diagnose():
    """Classify a leaf photo and return localized advice."""
    settings = _settings()
    language = normalize_language(request.form.get("language"))
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return _error(language, text(BAD_IMAGE, language), 400)

    data = upload.read()
    if not data:
        return _error(language, text(BAD_IMAGE, language), 400)
    if len(data) > settings.max_image_bytes:
        return _error(
            language,
            "The photo is too large. Please send a smaller image (under 8 MB).",
            400,
        )

    try:
        image = image_from_bytes(data)
    except ClassifierError:
        return _error(language, text(BAD_IMAGE, language), 400)

    classifier = _classifier()
    if not classifier.weights_exist():
        return _error(language, text(MISSING_MODEL, language), 503)

    try:
        predictions = classifier.predict(image, top_k=settings.top_k_predictions)
    except ClassifierError as exc:
        return _error(language, str(exc), 500)

    payload_preds = [
        {"label": item["label"], "confidence": item["confidence"]} for item in predictions
    ]
    top = predictions[0]
    note = disclaimer(language)

    if top["confidence"] < settings.confidence_threshold:
        advice = text(LOW_CONFIDENCE, language)
        return jsonify(
            {
                "status": "low_confidence",
                "predictions": payload_preds,
                "advice": with_disclaimer(advice, language),
                "language": language,
                "disclaimer": note,
            }
        )

    advice = generate_advice(
        display_label=top["label"],
        class_id=top["id"],
        kb_file=top["kb_file"],
        crop=top["crop"],
        language=language,
        settings=settings,
    )
    return jsonify(
        {
            "status": "ok",
            "predictions": payload_preds,
            "advice": with_disclaimer(advice, language),
            "language": language,
            "disclaimer": note,
        }
    )


def _error(language: str, message: str, status_code: int):
    body: dict[str, Any] = {
        "status": "error",
        "predictions": [],
        "advice": with_disclaimer(message, language),
        "language": language,
        "disclaimer": disclaimer(language),
    }
    return jsonify(body), status_code
