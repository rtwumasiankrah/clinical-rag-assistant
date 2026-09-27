#!/usr/bin/env python3
"""Ask the clinical RAG assistant a question from the command line.

Example:
    python scripts/ask.py "What is the protocol for managing sepsis in a critical care unit?"
"""
import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAGConfig
from src.qa import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the clinical RAG assistant.")
    parser.add_argument("question", help="Clinical question to answer.")
    parser.add_argument("--k", type=int, default=None, help="Retrieved chunks (default 3).")
    args = parser.parse_args()

    config = RAGConfig()
    if args.k:
        config.top_k = args.k

    result = answer_question(args.question, config)
    print("\nQUESTION:", result["question"])
    print("\nANSWER:\n" + textwrap.fill(result["answer"], width=88))
    pages = sorted({s["page"] for s in result["sources"] if s["page"]})
    print(f"\nSOURCES: manual pages {pages}")
    print(f"\nNOTE: {result['disclaimer']}")


if __name__ == "__main__":
    main()
