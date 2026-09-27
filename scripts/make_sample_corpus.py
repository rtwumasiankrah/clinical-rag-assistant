#!/usr/bin/env python3
"""Generate the small 2-page sample clinical corpus used for demos.

This is the same generator as the thin Colab walkthrough notebook
(notebooks/clinical_rag_walkthrough.ipynb) — extracted here so the Docker
entrypoint can bootstrap a demo index without duplicating the text.

Usage:
    python scripts/make_sample_corpus.py --out data/sample_manual.pdf
"""
import argparse
from pathlib import Path

SEPSIS = (
    "Sepsis is life-threatening organ dysfunction caused by a dysregulated host "
    "response to infection. Early recognition saves lives. For adults with suspected sepsis, "
    "obtain blood cultures before starting antibiotics, but do not delay antibiotics more than "
    "45 minutes to do so. Administer broad-spectrum antimicrobials within one hour of recognition. "
    "Begin rapid crystalloid fluid resuscitation for hypotension or serum lactate of 4 mmol/L or "
    "greater. Reassess frequently and narrow antibiotics once the pathogen is identified."
)

APPENDICITIS = (
    "Acute appendicitis is inflammation of the vermiform appendix, usually from luminal "
    "obstruction by a fecalith or lymphoid hyperplasia. Classic presentation begins with vague "
    "periumbilical pain that migrates to the right lower quadrant within 24 hours, accompanied by "
    "anorexia, nausea, and low-grade fever. Diagnosis is clinical, supported by an elevated white "
    "blood cell count and imaging such as ultrasound or CT. Definitive treatment is appendectomy; "
    "uncomplicated cases may be managed with antibiotics alone in selected patients."
)


def make_sample_corpus(out_path: Path) -> Path:
    """Write the 2-page sample PDF (Sepsis, Acute Appendicitis)."""
    import fitz  # PyMuPDF

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for title, body in [("Sepsis", SEPSIS), ("Acute Appendicitis", APPENDICITIS)]:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=18)
        page.insert_textbox(fitz.Rect(72, 110, 540, 720), body, fontsize=11)
    doc.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the 2-page sample clinical corpus PDF."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/sample_manual.pdf"),
        help="Where to write the sample PDF.",
    )
    args = parser.parse_args()
    out = make_sample_corpus(args.out)
    print(f"sample PDF written: {out}")


if __name__ == "__main__":
    main()
