#!/usr/bin/env python3
"""Build the Chroma vector index from the medical manual PDF (run once).

Example:
    python scripts/build_index.py --pdf data/medical_diagnosis_manual.pdf
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAGConfig
from src.ingest import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the clinical RAG vector index.")
    parser.add_argument("--pdf", type=Path, default=None, help="Path to the manual PDF.")
    parser.add_argument("--persist-dir", type=Path, default=None, help="Chroma persist directory.")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    args = parser.parse_args()

    config = RAGConfig()
    if args.pdf:
        config.pdf_path = args.pdf
    if args.persist_dir:
        config.persist_dir = args.persist_dir
    config.chunk_size = args.chunk_size
    config.chunk_overlap = args.chunk_overlap

    build_index(config)


if __name__ == "__main__":
    main()
