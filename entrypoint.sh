#!/usr/bin/env bash
# First-run bootstrap for the clinical-rag-assistant container, then exec uvicorn.
#
# On first run (no marker file in the data volume): generate the 2-page sample
# corpus, build the Chroma index, and let the Hugging Face models download into
# the volume (~5 GB total, cached for subsequent runs). Afterwards: start the API.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
PDF_PATH="${MEDICAL_PDF_PATH:-$DATA_DIR/sample_manual.pdf}"
PERSIST_DIR="${CHROMA_PERSIST_DIR:-$DATA_DIR/chroma}"
MARKER="$DATA_DIR/.index_ready"

mkdir -p "$DATA_DIR"

if [ ! -f "$MARKER" ]; then
  echo "[entrypoint] first run: generating sample corpus at $PDF_PATH"
  python /app/scripts/make_sample_corpus.py --out "$PDF_PATH"

  echo "[entrypoint] building Chroma index at $PERSIST_DIR"
  python /app/scripts/build_index.py \
    --pdf "$PDF_PATH" \
    --persist-dir "$PERSIST_DIR" \
    --chunk-size 200 \
    --chunk-overlap 20

  touch "$MARKER"
  echo "[entrypoint] bootstrap complete — index is ready"
else
  echo "[entrypoint] existing index found in $DATA_DIR — skipping bootstrap"
fi

echo "[entrypoint] starting API on :8000"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
