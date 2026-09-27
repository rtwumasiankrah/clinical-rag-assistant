"""HTTP API for the Clinical RAG Assistant (educational prototype).

Endpoints:
    GET  /health  -> {"status": "ok", "index_ready": bool}
    POST /ask     -> {"question": "..."} -> grounded answer with page sources

Nothing heavy loads at import time: the embedding model, the Chroma index, and
the Mistral-7B weights are all loaded lazily on the first /ask call (the
src.qa helpers cache them in module globals afterwards).

Run locally:
    uvicorn api.main:app --reload
Run in Docker:
    docker compose up --build
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.config import RAGConfig
from src.qa import answer_question

log = logging.getLogger("rag-api")

MAX_QUESTION_CHARS = 2000

_config = RAGConfig()
# llama-cpp's Llama object is not thread-safe; serialize /ask calls so the
# demo container can't crash under concurrent load.
_ask_lock = threading.Lock()


def index_ready() -> bool:
    """True when the Chroma collection exists and can be opened."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(_config.persist_dir))
        client.get_collection(_config.collection_name)
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "starting: pdf=%s chroma=%s collection=%s",
        _config.pdf_path,
        _config.persist_dir,
        _config.collection_name,
    )
    if index_ready():
        log.info("vector index found — /ask is ready")
    else:
        log.warning(
            "no vector index at %s — /ask will return 503 until one is built "
            "(in Docker this happens automatically on first run; otherwise run "
            "scripts/build_index.py)",
            _config.persist_dir,
        )
    yield
    log.info("shutting down")


app = FastAPI(
    title="Clinical RAG Assistant",
    description=(
        "Educational prototype: grounded medical Q&A over a clinical manual. "
        "Not medical advice."
    ),
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def _strip_and_check(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        if len(value) > MAX_QUESTION_CHARS:
            raise ValueError(
                f"question must be at most {MAX_QUESTION_CHARS} characters"
            )
        return value


class Source(BaseModel):
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    disclaimer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_ready": index_ready()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    if not index_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector index is not built yet. In Docker, the first-run "
                "bootstrap builds it automatically — check the container logs. "
                "Otherwise run: python scripts/build_index.py --pdf <manual.pdf>"
            ),
        )
    try:
        with _ask_lock:
            result = answer_question(request.question, _config)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — don't leak internals to callers
        log.exception("answer_question failed")
        raise HTTPException(
            status_code=500, detail="Failed to answer the question."
        ) from exc
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "disclaimer": result["disclaimer"],
    }
