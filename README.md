# ⚕️ Clinical RAG Assistant

[![CI](https://github.com/rtwumasiankrah/clinical-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/rtwumasiankrah/clinical-rag-assistant/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rtwumasiankrah/clinical-rag-assistant/blob/main/notebooks/clinical_rag_walkthrough.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Retrieval-Augmented Generation (RAG) over a **4,100-page medical reference manual** (the Merck Manual):
ask complex diagnostic, drug, and protocol questions and get answers grounded strictly in the text —
with page-level citations, a strict "I don't know" refusal policy, and LLM-judged quality scores.

> ⚠️ **Educational prototype — not medical advice.** Always consult a qualified clinician for
> diagnosis and treatment decisions.

## Why this exists

Clinicians face information overload: thousands of pages of trusted guidance, but no time to search
them during a critical-care decision. This project turns a static reference manual into a
**conversational clinical assistant** — the same idea as bringing expert-level medical intelligence
to clinics that can't keep a specialist on staff.

## How it works

**Pipeline:**

- **4,100-page PDF** → text extracted page by page with PyMuPDF.
- **Chunking** → token-aware chunks (512 tokens, 20-token overlap,
  `cl100k_base`) so context flows across chunk boundaries.
- **Embeddings** → every chunk embedded with `thenlper/gte-large`.
- **Chroma vector store** → persisted cosine-similarity index of chunk vectors.
- **Retrieval** → the question is embedded the same way; the top-3 most
  similar chunks are retrieved with page numbers attached.
- **Generation** → Mistral-7B-Instruct (quantized GGUF, run **locally** — no
  query text ever leaves the machine) answers **only from the retrieved
  context**, responding *"I don't know"* when the answer isn't there.
- **Evaluation** → the same LLM rates every answer as a judge on
  **groundedness** and **relevance**, each on a 1–5 rubric.

1. **Ingestion** — the manual is extracted with PyMuPDF and split into token-aware chunks
   (512 tokens, 20-token overlap, `cl100k_base`) so context flows across chunk boundaries.
2. **Embeddings** — every chunk is embedded with `thenlper/gte-large` and persisted in Chroma.
3. **Retrieval** — a question is embedded the same way; the top-3 most similar chunks are
   retrieved with page numbers attached.
4. **Generation** — Mistral-7B-Instruct (quantized GGUF, run **locally** — no query text ever
   leaves the machine) answers **only from the retrieved context**, and responds
   *"I don't know"* when the answer isn't there. A refused answer beats a hallucinated one.
5. **Evaluation** — the same LLM rates every answer as a judge on **groundedness**
   (is it derived only from the context?) and **relevance** (does it address the question?),
   each on a 1–5 rubric.

## Quickstart

```bash
git clone https://github.com/rtwumasiankrah/clinical-rag-assistant.git
cd clinical-rag-assistant
pip install -r requirements.txt

# 1. add your copy of the manual (not included — see data/README.md)
# 2. build the vector index (once)
python scripts/build_index.py --pdf data/medical_diagnosis_manual.pdf

# 3. ask a question
python scripts/ask.py "What is the protocol for managing sepsis in a critical care unit?"

# or launch the demo UI
streamlit run app/streamlit_app.py
```

The Mistral-7B weights (~4 GB GGUF) download automatically from the Hugging Face Hub on first run.

## Run with Docker

The API is containerized — one command builds the image and starts the service:

```bash
docker compose up --build
```

**What happens on first run:** the entrypoint bootstraps everything automatically —
it generates the 2-page sample corpus (the same one the Colab walkthrough uses),
builds the Chroma index, and downloads the models into a named Docker volume:
the Mistral-7B GGUF weights (~4 GB) and the `thenlper/gte-large` embedding model
(~1.3 GB). Expect the first run to take a while (large downloads plus the index
build); subsequent runs reuse the volume and start in seconds. Nothing
model-sized is baked into the image.

**Endpoints** (once the logs show the API listening on :8000):

```bash
# health + index readiness
curl http://localhost:8000/health

# ask a question (grounded in the sample corpus: sepsis & appendicitis)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended timing for antibiotics in sepsis?"}'
```

`POST /ask` returns `{"answer", "sources": [{"page"}], "disclaimer"}` and answers
`503` while the index is still building. As with the rest of this repo, the
container serves an **educational prototype — not medical advice** (every
response carries the disclaimer).

## What's in this repo

| Path | What it is |
|---|---|
| `notebooks/medical_diagnosis_rag.ipynb` | The original coursework notebook, as run — 140 cells covering prompt engineering, RAG, and evaluation |
| `notebooks/clinical_rag_walkthrough.ipynb` | Thin Colab walkthrough — calls `scripts/` and `src/` directly on a sample corpus, runs end-to-end in the browser |
| `src/` | Productionized pipeline: `config.py`, `chunking.py`, `ingest.py`, `qa.py`, `evaluate.py`, `prompts.py` (no LangChain dependency) |
| `scripts/build_index.py` | One-shot CLI: PDF → Chroma index |
| `scripts/ask.py` | CLI: ask a question, get a cited answer |
| `scripts/make_sample_corpus.py` | Generates the 2-page sample corpus (mirrors the walkthrough notebook) |
| `api/main.py` | FastAPI service: `POST /ask`, `GET /health` (lazy model loading) |
| `Dockerfile`, `docker-compose.yml`, `entrypoint.sh` | Containerized deployment; first-run bootstrap into a named volume |
| `app/streamlit_app.py` | Interactive demo with retrieved-evidence viewer |
| `tests/` | Unit tests for chunking, prompts, and config (run in CI) |
| `data/README.md` | How to obtain the manual PDF |

## Results

Evaluated end-to-end on clinical questions (sepsis protocol, appendicitis, alopecia areata,
traumatic brain injury, leg fracture):

- **Groundedness & relevance:** LLM-judged 5/5 on tested queries — answers stayed inside the
  retrieved context with no fabricated detail.
- **Prompt engineering mattered:** a strict system prompt ("answer only using the context…
  if the answer is not found, respond *I don't know*") eliminated the generic, ungrounded
  answers the base model produced before RAG.
- **Retrieval tuning mattered:** increasing `k` helped when answers spanned multiple manual
  sections; chunk overlap preserved context across segment boundaries; `max_tokens` and
  temperature were calibrated per query complexity at temperature 0 for determinism.

Key engineering takeaways: index-build time scales with corpus size; retrieval depth (`k`)
is the highest-leverage parameter; and in clinical settings, evaluation must weight
*groundedness* above fluency.

## Limitations & ethics

- Prototype-scale evaluation — a production system needs clinician-reviewed benchmarks
  (e.g., MedQA-style) and human oversight, not just LLM judges.
- The manual is a snapshot in time; clinical guidance evolves, so any deployment needs a
  re-indexing and versioning story.
- This tool supports decisions; it does not make them.

## Roadmap

- Swap in domain-tuned biomedical embeddings and re-rankers (e.g., cross-encoders).
- Add citation-level faithfulness checks (e.g., entailment scoring per claim).
- **Sequence modeling for genomics:** the same transformer architectures used here for text
  transfer directly to biological sequences — a planned follow-up applies sequence models
  (CNN/Transformer) to viral genomic data for mutation-trend analysis, connecting this work
  to phage genomics research.

## Author

**Richard Twumasi-Ankrah** — M.S. Computer Science, building AI that makes medical
intelligence widely accessible. [GitHub](https://github.com/rtwumasiankrah)
