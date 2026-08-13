# DSP Palette

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20092215.svg)](https://doi.org/10.5281/zenodo.20092215)

Code, evaluation data, and a public reference implementation for the method.

## Implementation

You can try the live implementation of DSP Palette at [Véridique](https://qide.studio/tools/veridique/veridique.html).

> Michail Semoglou, "Distinctness-First Palette Extraction for Accessible Design Systems," accepted to _APSIPA ASC 2026_, Track IVM.

## Project status

- Accepted paper: APSIPA ASC 2026, Track IVM
- Public research artifact: reproducible code, evaluation pipeline, and dataset split
- Concept DOI (all versions): 10.5281/zenodo.20092215
- Latest release DOI (v1.1.0): 10.5281/zenodo.21915330

## Abstract

Design systems need color palettes that are perceptually distinct and carry a guaranteed WCAG AA-compliant Surface/On-Surface pair: k-means clustering and median-cut quantization cannot meet these requirements because they optimize solely for reconstruction fidelity. This paper introduces **Distinctness-First Palette Selection (DSP)**, a constrained greedy algorithm that selects _n_ colors by maximizing minimum pairwise ΔE₂₀₀₀ (perceptual distinctness) while enforcing a hard inter-color separation threshold τ and a post-selection WCAG AA contrast check. A heuristic assigns each color a semantic role (Surface, On-Surface, Primary, Secondary, Accent) suited to contemporary design-token schemas.

Evaluated on a held-out test set of _N_ = 75 COCO photographs under four metrics, DSP achieves a mean minimum ΔE₂₀₀₀ of **24.9 ± 5.6** against 14.1 ± 4.9, 12.4 ± 3.8, and 7.2 ± 3.9 for k-Means Lab, k-Means RGB, and Median Cut respectively (all _p_ < 0.001, Wilcoxon signed-rank; Cliff's δ > 0.84, large effect in every case). WCAG AA coverage improves significantly over k-Means Lab and Median Cut (both _p_ < 0.004, Bonferroni-corrected); the gain over k-Means RGB is nominally significant (_p_ = 0.013) but does not survive Bonferroni correction. Rankings hold on a disjoint 25-image COCO train2017 set. Reconstruction error is higher by design: the method trades pixel-level fidelity for perceptual spread. DSP is available as open-source software with a persistent DOI and a reproducible evaluation corpus.

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
dsp/                  # DSP method implementation (selector, roles, and metrics)
baselines/            # k-Means Lab, k-Means RGB, Median Cut, and ColorThief
evaluation/           # evaluation runner, metrics, and aggregation scripts
corpus/
  manifest.json       # image IDs with dev/test split labels
  download.py         # script to fetch COCO images
results/
  raw/                # per-image evaluation JSONs (N = 115)
  aggregated/         # CSV summaries and summary_for_paper.md
tables/               # standalone LaTeX table sources (Tables 1–3)
figures/              # pipeline diagnostic figures
tests/                # unit tests
```

## Reproducing the Evaluation

1. Download the corpus images (COCO val2017/train2017, CC BY 4.0):

   ```bash
   python corpus/download.py
   ```

2. Run the evaluation:

   ```bash
   python -m evaluation.runner
   ```

3. Aggregate results and regenerate figures:
   ```bash
   python -m evaluation.report
   ```

Pre-computed results are already included in `results/` for inspection without re-running.

## Running Tests

```bash
pytest tests/
```

## Citation

If you use this code or data, please cite:

```bibtex
@software{semoglou_dsp_palette_2026,
  author    = {Semoglou, Michail},
  title     = {{DSP Palette: Distinctness-First Palette Extraction for Accessible Design Systems}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.1.0},
  doi       = {10.5281/zenodo.20092215},
  url       = {https://doi.org/10.5281/zenodo.20092215}
}
```

## License

MIT — see [LICENSE](LICENSE).
