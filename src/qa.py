"""Grounded clinical Q&A: retrieve context, then generate with Mistral-7B.

The LLM runs locally via llama-cpp-python (GGUF weights downloaded once from
the Hugging Face Hub), so no patient-like query text ever leaves the machine.
"""
from .config import RAGConfig
from .ingest import get_embedding_model, load_index
from .prompts import MEDICAL_DISCLAIMER, build_qa_prompt

_llm = None


def get_llm(config: RAGConfig):
    """Lazily download (first run) and load the GGUF instruction model."""
    global _llm
    if _llm is None:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        model_path = hf_hub_download(
            repo_id=config.llm_repo_id, filename=config.llm_filename
        )
        _llm = Llama(
            model_path=model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False,
        )
    return _llm


def retrieve_context(question: str, config: RAGConfig) -> list:
    """Return the top-k most relevant chunks for *question* with provenance."""
    collection = load_index(config)
    model = get_embedding_model(config)
    query_embedding = model.encode([question], convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=config.retrieval_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "page": meta.get("page"), "distance": dist})
    return hits


def answer_question(question: str, config: RAGConfig | None = None) -> dict:
    """Answer a clinical question strictly from the retrieved manual context."""
    config = config or RAGConfig()
    hits = retrieve_context(question, config)
    context = ". ".join(h["text"] for h in hits)
    prompt = build_qa_prompt(context=context, question=question)

    llm = get_llm(config)
    response = llm(
        prompt=prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )
    answer = response["choices"][0]["text"].strip()
    return {
        "question": question,
        "answer": answer,
        "sources": [{"page": h["page"]} for h in hits],
        "disclaimer": MEDICAL_DISCLAIMER,
    }
