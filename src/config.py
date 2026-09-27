"""Central configuration for the clinical RAG pipeline.

Every value can be overridden with an environment variable so the pipeline
runs unchanged on a laptop, a Colab runtime, or a server.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class RAGConfig:
    """Tunable parameters for ingestion, retrieval, and generation."""

    # --- corpus ---
    pdf_path: Path = field(
        default_factory=lambda: Path(
            _env("MEDICAL_PDF_PATH", str(PROJECT_ROOT / "data" / "medical_diagnosis_manual.pdf"))
        )
    )
    persist_dir: Path = field(
        default_factory=lambda: Path(
            _env("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "medical_db"))
        )
    )
    collection_name: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "merck_manual"))

    # --- chunking (token-aware, cl100k_base) ---
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "20")))

    # --- embeddings ---
    embedding_model_name: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "thenlper/gte-large")
    )

    # --- retrieval ---
    retrieval_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_K", "3")))

    # --- generation (Mistral-7B-Instruct GGUF via llama-cpp-python) ---
    llm_repo_id: str = field(
        default_factory=lambda: _env("LLM_REPO_ID", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    )
    llm_filename: str = field(
        default_factory=lambda: _env("LLM_FILENAME", "mistral-7b-instruct-v0.2.Q6_K.gguf")
    )
    n_ctx: int = field(default_factory=lambda: int(_env("LLM_N_CTX", "4096")))
    n_gpu_layers: int = field(default_factory=lambda: int(_env("LLM_N_GPU_LAYERS", "0")))
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "500")))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.0")))
    top_p: float = field(default_factory=lambda: float(_env("LLM_TOP_P", "0.95")))
    top_k: int = field(default_factory=lambda: int(_env("LLM_TOP_K", "50")))
