"""CropSense Flask application factory."""

from __future__ import annotations

from flask import Flask

from app.config import load_settings
from app.routes import bp


def create_app() -> Flask:
    """Create and configure the Flask app."""
    settings = load_settings()
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config["MAX_CONTENT_LENGTH"] = settings.max_image_bytes
    app.config["CROPSENSE_SETTINGS"] = settings
    app.register_blueprint(bp)
    return app
