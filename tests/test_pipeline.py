"""Unit tests for the model-free parts of the pipeline.

These run in CI without downloading the LLM, the embedding model, or the
manual — they cover chunking, prompts, and configuration.
"""
import os
import sys
from pathlib import Path

import pytest
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_pages, split_text
from src.config import RAGConfig
from src.prompts import (
    MEDICAL_DISCLAIMER,
    QNA_SYSTEM_MESSAGE,
    build_eval_prompt,
    build_qa_prompt,
)

ENC = tiktoken.get_encoding("cl100k_base")


def test_split_text_respects_token_budget():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = split_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(ENC.encode(c)) <= 100


def test_split_text_overlap_preserves_context():
    text = "Alpha. " * 40 + "MARKER " + "Beta. " * 40
    chunks = split_text(text, chunk_size=30, chunk_overlap=10)
    assert any("MARKER" in c for c in chunks)
    # overlapping windows should share tokens between neighbours
    assert len(chunks) >= 2


def test_split_text_empty_input():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_split_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("hello world", chunk_size=10, chunk_overlap=10)


def test_chunk_pages_keeps_provenance():
    pages = [(3, "Fever is a common symptom. " * 50), (7, "Aspirin reduces fever. " * 50)]
    chunks = chunk_pages(pages, chunk_size=40, chunk_overlap=5)
    assert chunks
    assert {c.page for c in chunks} == {3, 7}
    assert all(c.text.strip() for c in chunks)


def test_qa_prompt_is_grounded():
    prompt = build_qa_prompt(context="Sepsis protocol: give broad-spectrum antibiotics.",
                             question="How is sepsis managed?")
    assert "Sepsis protocol" in prompt
    assert "How is sepsis managed?" in prompt
    assert "I don't know" in prompt  # refusal instruction present
    assert "only using the context" in prompt


def test_eval_prompt_structure():
    prompt = build_eval_prompt("SYSTEM", "Q?", "CTX", "ANS")
    assert prompt.startswith("[INST]SYSTEM")
    assert "###Question\nQ?" in prompt
    assert "###Context\nCTX" in prompt
    assert "###Answer\nANS" in prompt


def test_config_defaults_match_coursework():
    config = RAGConfig()
    assert config.chunk_size == 512
    assert config.chunk_overlap == 20
    assert config.top_k == 3
    assert config.max_tokens == 500
    assert config.temperature == 0.0
    assert "gte-large" in config.embedding_model_name
    assert "Mistral-7B-Instruct" in config.llm_repo_id


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("RETRIEVAL_K", "5")
    config = RAGConfig()
    assert config.chunk_size == 256
    assert config.top_k == 5


def test_disclaimer_present():
    assert "not medical advice" in MEDICAL_DISCLAIMER
    assert "clinician" in MEDICAL_DISCLAIMER
    assert QNA_SYSTEM_MESSAGE  # system prompt is non-empty
