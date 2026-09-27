"""LLM-as-judge evaluation: groundedness + relevance on a 1-5 scale.

The same local LLM that answers questions also rates them, using structured
rubrics. This gives a cheap, repeatable quality signal without human labels —
a pragmatic stand-in for clinician review during prototyping.
"""
import re

from .config import RAGConfig
from .prompts import (
    GROUNDEDNESS_RATER_SYSTEM_MESSAGE,
    RELEVANCE_RATER_SYSTEM_MESSAGE,
    build_eval_prompt,
)
from .qa import answer_question, get_llm, retrieve_context


def _extract_score(rating_text: str) -> int | None:
    """Pull the first 1-5 rating out of a free-text judge response."""
    match = re.search(r"\b([1-5])\b", rating_text)
    return int(match.group(1)) if match else None


def rate_answer(question: str, context: str, answer: str, config: RAGConfig) -> dict:
    """Score one answer on groundedness and relevance (1-5 each)."""
    llm = get_llm(config)
    scores = {}
    for metric, system_message in (
        ("groundedness", GROUNDEDNESS_RATER_SYSTEM_MESSAGE),
        ("relevance", RELEVANCE_RATER_SYSTEM_MESSAGE),
    ):
        prompt = build_eval_prompt(system_message, question, context, answer)
        response = llm(
            prompt=prompt,
            max_tokens=256,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=["[/INST]"],
        )
        text = response["choices"][0]["text"].strip()
        scores[metric] = {"score": _extract_score(text), "rationale": text}
    return scores


def evaluate_questions(questions: list, config: RAGConfig | None = None) -> list:
    """End-to-end evaluation: answer each question, then judge it."""
    config = config or RAGConfig()
    report = []
    for question in questions:
        result = answer_question(question, config)
        hits = retrieve_context(question, config)
        context = ". ".join(h["text"] for h in hits)
        ratings = rate_answer(question, context, result["answer"], config)
        report.append(
            {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "groundedness": ratings["groundedness"]["score"],
                "relevance": ratings["relevance"]["score"],
            }
        )
    return report


DEFAULT_EVAL_QUESTIONS = [
    "What is the protocol for managing sepsis in a critical care unit?",
    "What are the typical clinical symptoms of appendicitis, and when is surgery indicated?",
    "What are the effective treatments for sudden-onset patchy hair loss (alopecia areata)?",
    "What are the recommended acute and long-term treatments for traumatic brain injury?",
    "Outline the first aid, diagnostic steps, and treatment plan for a leg fracture.",
]
