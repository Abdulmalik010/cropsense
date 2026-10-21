"""Advice generator must ground treatment text in retrieved passages."""

from app.advisor import generate_advice
from app.messages import NO_KNOWLEDGE


def test_generate_advice_uses_only_passages(monkeypatch):
    captured = {}

    def fake_retrieve(class_id, kb_file, crop, settings=None):
        return [
            "Early blight makes target-like brown spots on older tomato leaves. "
            "Water the soil, not the leaves. Do not guess a pesticide name."
        ]

    def fake_post(url, json=None, timeout=None):
        captured["prompt"] = json["prompt"]
        captured["model"] = json["model"]

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"response": "Early blight makes target-like spots. Water the soil."}

        return _Resp()

    monkeypatch.setattr("app.advisor.retrieve_passages", fake_retrieve)
    monkeypatch.setattr("app.advisor.requests.post", fake_post)

    advice = generate_advice(
        display_label="Tomato Early Blight",
        class_id="Tomato___Early_blight",
        kb_file="tomato-early-blight.md",
        crop="tomato",
        language="en",
    )
    assert "target-like" in advice.lower() or "early blight" in advice.lower()
    prompt = captured["prompt"]
    assert "Never invent pesticide" in prompt
    assert "ONLY the knowledge-base passages" in prompt
    assert "mancozeb" not in prompt.lower()
    assert captured["model"]


def test_generate_advice_without_passages(monkeypatch):
    monkeypatch.setattr("app.advisor.retrieve_passages", lambda *args, **kwargs: [])
    advice = generate_advice(
        display_label="Tomato Early Blight",
        class_id="Tomato___Early_blight",
        kb_file="tomato-early-blight.md",
        crop="tomato",
        language="en",
    )
    assert advice == NO_KNOWLEDGE["en"]
    assert "pesticide" in advice.lower()
