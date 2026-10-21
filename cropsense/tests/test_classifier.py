"""Classifier helpers that do not need trained weights."""

import pytest

from app.classifier import ClassifierError, CropClassifier, image_from_bytes
from tests.conftest import rgb_jpeg_bytes


def test_image_from_bytes_accepts_jpeg():
    image = image_from_bytes(rgb_jpeg_bytes())
    assert image.mode == "RGB"
    assert image.size == (64, 64)


def test_image_from_bytes_rejects_garbage():
    with pytest.raises(ClassifierError):
        image_from_bytes(b"not-an-image")


def test_missing_weights_status(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASSIFIER_WEIGHTS", str(tmp_path / "missing.pt"))
    from app.config import load_settings

    clf = CropClassifier(load_settings())
    assert clf.weights_exist() is False
    with pytest.raises(ClassifierError):
        clf.load()
