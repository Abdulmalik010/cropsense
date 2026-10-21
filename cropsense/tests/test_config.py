"""Config and label catalog tests."""

from app.config import SUPPORTED_LANGUAGES, class_by_id, class_ids, load_label_catalog, load_settings
from app.messages import LOW_CONFIDENCE, normalize_language


def test_settings_read_defaults(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "gemma4:e4b")
    settings = load_settings()
    assert settings.llm_model == "gemma4:e4b"
    assert settings.confidence_threshold == 0.6
    assert settings.ollama_host.startswith("http")


def test_labels_cover_four_crops():
    ids = class_ids()
    catalog = load_label_catalog()
    crops = {item["crop"] for item in catalog["classes"]}
    assert crops == {"tomato", "maize", "cassava", "pepper"}
    assert "Tomato___Early_blight" in ids
    meta = class_by_id("Pepper___Bacterial_spot")
    assert meta["kb_file"] == "pepper-bacterial-spot.md"


def test_language_fallback():
    assert normalize_language("yo") == "yo"
    assert normalize_language("fr") == "en"
    assert "pcm" in SUPPORTED_LANGUAGES
    assert "photo" in LOW_CONFIDENCE["en"].lower()
    assert "extension officer" in LOW_CONFIDENCE["en"].lower()
