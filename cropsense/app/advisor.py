"""Retrieve knowledge-base passages and draft advice with Ollama."""

from __future__ import annotations

from typing import Any

import requests

from app.config import SUPPORTED_LANGUAGES, Settings, load_settings
from app.messages import ADVISOR_OFFLINE, NO_KNOWLEDGE, disclaimer, text

COLLECTION_NAME = "cropsense_kb"


class AdvisorError(Exception):
    """Raised when retrieval or generation fails."""


def ollama_alive(settings: Settings | None = None) -> dict[str, Any]:
    """Check whether the Ollama HTTP API responds."""
    cfg = settings or load_settings()
    url = f"{cfg.ollama_host}/api/tags"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        names = [item.get("name", "") for item in response.json().get("models", [])]
        return {
            "ready": True,
            "host": cfg.ollama_host,
            "llm_present": any(cfg.llm_model in name for name in names),
            "embed_present": any(cfg.embed_model in name for name in names),
            "models": names,
        }
    except requests.RequestException as exc:
        return {
            "ready": False,
            "host": cfg.ollama_host,
            "error": str(exc),
        }


class OllamaEmbeddingFunction:
    """Chroma embedding function backed by the Ollama embed API."""

    def __init__(self, host: str, model: str) -> None:
        self.host = host.rstrip("/")
        self.model = model

    def name(self) -> str:
        return f"ollama/{self.model}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text_item in input:
            vectors.append(embed_text(self.host, self.model, text_item))
        return vectors


def embed_text(host: str, model: str, prompt: str) -> list[float]:
    """Return one embedding vector from Ollama."""
    url = f"{host.rstrip('/')}/api/embed"
    try:
        response = requests.post(
            url, json={"model": model, "input": prompt}, timeout=60
        )
        if response.status_code == 404:
            response = requests.post(
                f"{host.rstrip('/')}/api/embeddings",
                json={"model": model, "prompt": prompt},
                timeout=60,
            )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise AdvisorError(f"Ollama embeddings failed: {exc}") from exc
    if "embeddings" in payload:
        return payload["embeddings"][0]
    if "embedding" in payload:
        return payload["embedding"]
    raise AdvisorError("Ollama embed response missing embedding vector.")


def _collection(settings: Settings):
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    embed_fn = OllamaEmbeddingFunction(settings.ollama_host, settings.embed_model)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_passages(
    class_id: str,
    kb_file: str,
    crop: str,
    settings: Settings | None = None,
) -> list[str]:
    """Fetch the most relevant knowledge-base chunks for a predicted class."""
    cfg = settings or load_settings()
    collection = _collection(cfg)
    if collection.count() == 0:
        return []
    query = f"{class_id} {crop} {kb_file} symptoms treatment prevention"
    result = collection.query(
        query_texts=[query],
        n_results=cfg.rag_n_results,
        where={"kb_file": kb_file},
    )
    documents = result.get("documents") or []
    if not documents or not documents[0]:
        result = collection.query(query_texts=[query], n_results=cfg.rag_n_results)
        documents = result.get("documents") or []
    return [chunk for chunk in (documents[0] if documents else []) if chunk]


def generate_advice(
    *,
    display_label: str,
    class_id: str,
    kb_file: str,
    crop: str,
    language: str,
    settings: Settings | None = None,
) -> str:
    """Write short advice in `language` using only retrieved passages."""
    cfg = settings or load_settings()
    language_name = SUPPORTED_LANGUAGES.get(language, "English")
    try:
        passages = retrieve_passages(class_id, kb_file, crop, cfg)
    except AdvisorError:
        return text(ADVISOR_OFFLINE, language)
    if not passages:
        return text(NO_KNOWLEDGE, language)

    numbered = "\n\n".join(f"[{index + 1}] {chunk}" for index, chunk in enumerate(passages))
    prompt = (
        f"You are CropSense, a crop-disease advisor for smallholder farmers in Nigeria.\n"
        f"Write in {language_name} only.\n"
        f"The image classifier predicted: {display_label} ({class_id}) on crop {crop}.\n"
        "Explain what the disease is (or that the plant looks healthy), practical "
        "treatment options, and prevention.\n"
        "Use ONLY the knowledge-base passages below.\n"
        "Never invent pesticide names, active ingredients, doses, mixing rates, or "
        "spray schedules that are not written in the passages.\n"
        "If the passages do not name a product, tell the farmer to ask a local "
        "extension officer for products registered in their state.\n"
        "Keep the answer short (about 150-220 words). Use simple sentences.\n"
        "Do not add a disclaimer; the app will append one.\n\n"
        f"PASSAGES:\n{numbered}\n"
    )
    try:
        response = requests.post(
            f"{cfg.ollama_host}/api/generate",
            json={
                "model": cfg.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        response.raise_for_status()
        advice = (response.json().get("response") or "").strip()
    except requests.RequestException:
        return text(ADVISOR_OFFLINE, language)
    if not advice:
        return text(ADVISOR_OFFLINE, language)
    return advice


def chroma_status(settings: Settings | None = None) -> dict[str, Any]:
    """Describe knowledge-base index availability for `/health`."""
    cfg = settings or load_settings()
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(cfg.chroma_path))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        return {"ready": collection.count() > 0, "chunks": collection.count()}
    except Exception as exc:  # noqa: BLE001 — health must never crash
        return {"ready": False, "error": str(exc)}


def with_disclaimer(advice: str, language: str) -> str:
    """Append the required decision-support disclaimer."""
    note = disclaimer(language)
    if note in advice:
        return advice
    return f"{advice.rstrip()}\n\n{note}"
