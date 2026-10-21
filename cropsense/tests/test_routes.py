"""API behaviour for diagnose and health."""

from io import BytesIO

from tests.conftest import rgb_jpeg_bytes


class _FakeClassifier:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def weights_exist(self) -> bool:
        return True

    def predict(self, image, top_k=None):
        return [
            {
                "id": "Tomato___Early_blight",
                "label": "Tomato Early Blight",
                "crop": "tomato",
                "kb_file": "tomato-early-blight.md",
                "confidence": self.confidence,
            },
            {
                "id": "Tomato___Late_blight",
                "label": "Tomato Late Blight",
                "crop": "tomato",
                "kb_file": "tomato-late-blight.md",
                "confidence": 0.2,
            },
            {
                "id": "Tomato___healthy",
                "label": "Tomato (healthy)",
                "crop": "tomato",
                "kb_file": "tomato-healthy.md",
                "confidence": 0.1,
            },
        ]


def test_health_json(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert "classifier" in body
    assert "ollama" in body
    assert body["llm_model"]


def test_index_has_camera_capture(client):
    response = client.get("/")
    html = response.get_data(as_text=True)
    assert 'accept="image/*"' in html
    assert 'capture="environment"' in html
    assert "Hausa" in html


def test_diagnose_rejects_empty(client):
    response = client.post("/api/diagnose", data={"language": "en"})
    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert body["disclaimer"]


def test_low_confidence_does_not_call_advisor(client, app, monkeypatch):
    called = {"advisor": False}

    def boom(**kwargs):
        called["advisor"] = True
        raise AssertionError("advisor should not run on low confidence")

    monkeypatch.setattr("app.routes.generate_advice", boom)
    app.config["CROPSENSE_CLASSIFIER"] = _FakeClassifier(0.21)

    response = client.post(
        "/api/diagnose",
        data={"language": "en", "image": (BytesIO(rgb_jpeg_bytes()), "leaf.jpg")},
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "low_confidence"
    assert called["advisor"] is False
    assert "extension" in body["advice"].lower()
    assert body["disclaimer"]
    assert len(body["predictions"]) == 3


def test_ok_diagnose_uses_advisor(client, app, monkeypatch):
    monkeypatch.setattr(
        "app.routes.generate_advice",
        lambda **kwargs: "Water the soil, not the leaves.",
    )
    app.config["CROPSENSE_CLASSIFIER"] = _FakeClassifier(0.91)
    response = client.post(
        "/api/diagnose",
        data={"language": "ha", "image": (BytesIO(rgb_jpeg_bytes()), "leaf.jpg")},
    )
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["language"] == "ha"
    assert body["predictions"][0]["label"] == "Tomato Early Blight"
    assert "Water the soil" in body["advice"]
    assert body["disclaimer"]
