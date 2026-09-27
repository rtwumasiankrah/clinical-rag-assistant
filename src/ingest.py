"""Corpus ingestion: PDF -> token-aware chunks -> embeddings -> Chroma.

Run once per manual:

    python scripts/build_index.py --pdf data/medical_diagnosis_manual.pdf
"""
from pathlib import Path

import chromadb

from .chunking import chunk_pages
from .config import RAGConfig

_embedding_model = None


def get_embedding_model(config: RAGConfig):
    """Lazily load the sentence-transformer embedding model (cached)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(config.embedding_model_name)
    return _embedding_model


def load_pdf_pages(pdf_path: Path) -> list:
    """Extract ``(page_number, text)`` pairs from a PDF with PyMuPDF."""
    import fitz

    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append((i, text))
    return pages


def build_index(config: RAGConfig):
    """Chunk, embed, and persist the manual into a Chroma collection."""
    if not config.pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {config.pdf_path}. See data/README.md for how to obtain it."
        )

    print(f"Loading {config.pdf_path} ...")
    pages = load_pdf_pages(config.pdf_path)
    print(f"  {len(pages)} pages with extractable text")

    print("Chunking ...")
    chunks = chunk_pages(
        pages, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    print(f"  {len(chunks)} chunks")

    print(f"Embedding with {config.embedding_model_name} ...")
    model = get_embedding_model(config)
    embeddings = model.encode(
        [c.text for c in chunks], show_progress_bar=True, convert_to_numpy=True
    ).tolist()

    print(f"Persisting Chroma collection to {config.persist_dir} ...")
    client = chromadb.PersistentClient(path=str(config.persist_dir))
    # Rebuilding from scratch keeps the demo reproducible.
    try:
        client.delete_collection(config.collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.collection_name, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"page": c.page, "chunk_index": c.chunk_index} for c in chunks],
    )
    print(f"  done: {collection.count()} vectors indexed")
    return collection


def load_index(config: RAGConfig):
    """Open an already-built Chroma collection (read path for Q&A)."""
    client = chromadb.PersistentClient(path=str(config.persist_dir))
    return client.get_collection(config.collection_name)
