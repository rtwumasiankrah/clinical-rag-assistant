# Data

This project was built against a ~4,100-page medical reference manual
(the Merck Manual). The PDF itself is **not included in this repository**
(it is copyrighted material, ~20 MB).

To run the pipeline yourself:

1. Obtain a copy of the manual PDF.
2. Place it at `data/medical_diagnosis_manual.pdf`
   (or set the `MEDICAL_PDF_PATH` environment variable to its location).
3. Build the vector index once:
   ```bash
   python scripts/build_index.py --pdf data/medical_diagnosis_manual.pdf
   ```

The pipeline also works with any long-form medical reference PDF —
a drug formulary, clinical protocols handbook, or similar.
