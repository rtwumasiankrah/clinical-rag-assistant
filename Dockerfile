# Clinical RAG Assistant — containerized FastAPI service (educational prototype).
#
# The image contains code + Python deps only. The ~4 GB Mistral-7B weights, the
# ~1.3 GB embedding model, and the Chroma index all live in the /data volume so
# they download once on first run and persist across restarts. They are NOT
# baked into the image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache \
    MEDICAL_PDF_PATH=/data/sample_manual.pdf \
    CHROMA_PERSIST_DIR=/data/chroma

# llama-cpp-python falls back to a source build when no prebuilt wheel matches,
# so keep a compiler around; cmake is required by the llama.cpp build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: keeps the image lean (the default torch wheel bundles CUDA).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Prebuilt CPU wheel for llama-cpp-python (avoids a ~10 min source compile).
RUN pip install --no-cache-dir llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY data/README.md ./data/README.md
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
