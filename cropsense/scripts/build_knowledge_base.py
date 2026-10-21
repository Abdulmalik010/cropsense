"""Chunk markdown knowledge files and store embeddings in ChromaDB."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.advisor import COLLECTION_NAME, OllamaEmbeddingFunction, ollama_alive
from app.config import load_label_catalog, load_settings


def chunk_markdown(text: str, max_chars: int = 700) -> list[str]:
    """Split a markdown file into small retrieval chunks."""
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > max_chars:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}".strip()
    if current:
        chunks.append(current)
    return chunks


def main() -> None:
    settings = load_settings()
    status = ollama_alive(settings)
    if not status.get("ready"):
        raise SystemExit(
            "Ollama is not reachable. Start it, then pull "
            f"`{settings.embed_model}` before building the knowledge base."
        )
    if not status.get("embed_present"):
        raise SystemExit(
            f"Embedding model `{settings.embed_model}` is not on the Ollama host. "
            f"Run: ollama pull {settings.embed_model}"
        )

    catalog = load_label_catalog(settings.labels_path)
    kb_files = {item["kb_file"]: item for item in catalog["classes"]}
    knowledge_dir = settings.knowledge_dir
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddingFunction(settings.ollama_host, settings.embed_model),
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    for path in sorted(knowledge_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        meta = kb_files.get(path.name)
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_markdown(text)):
            ids.append(f"{path.stem}-{index}")
            documents.append(chunk)
            metadatas.append(
                {
                    "kb_file": path.name,
                    "class_id": meta["id"] if meta else path.stem,
                    "crop": meta["crop"] if meta else "unknown",
                }
            )

    if not documents:
        raise SystemExit(f"No markdown files found in {knowledge_dir}")

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(documents)} chunks from {knowledge_dir} -> {settings.chroma_path}")


if __name__ == "__main__":
    main()
