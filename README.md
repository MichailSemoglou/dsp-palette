# DSP Palette

Code and evaluation data for the paper:

> Michail Semoglou, "Distinctness-First Palette Extraction for Accessible Design Systems," submitted to _APSIPA ASC 2026_, Track IVM.

## Abstract

Design systems need color palettes that are perceptually distinct and WCAG-accessible by construction — requirements that k-means clustering and median-cut quantization cannot meet because they optimize solely for reconstruction fidelity. This paper introduces **Distinctness-First Palette Selection (DSP)**, a constrained greedy algorithm that selects _n_ colors by jointly maximizing perceptual coverage (log-frequency and minimum ΔE₂₀₀₀) while enforcing a hard inter-color separation threshold τ and a post-selection WCAG AA contrast check. A heuristic assigns each color a semantic role (Surface, On-Surface, Primary, Secondary, Accent) suited to contemporary design-token schemas.

Evaluated on a held-out test set of _N_ = 75 COCO photographs, DSP achieves a mean minimum ΔE₂₀₀₀ of **24.9 ± 5.6** against 14.1 ± 4.9, 12.4 ± 3.8, and 7.2 ± 3.9 for k-Means Lab, k-Means RGB, and Median Cut respectively (all _p_ < 0.001, Wilcoxon signed-rank; Cliff's δ > 0.84, large effect in every case).

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```
dsp/                  # DSP method implementation (selector, roles, metrics)
baselines/            # k-Means Lab, k-Means RGB, Median Cut, ColorThief
evaluation/           # evaluation runner, metrics, aggregation scripts
corpus/
  manifest.json       # image IDs with dev/test split labels
  download.py         # script to fetch COCO images
results/
  raw/                # per-image evaluation JSONs (N = 115)
  aggregated/         # CSV summaries used in the paper
  summary_for_paper.md
figures/              # PDF figures and LaTeX table sources
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

## License

MIT — see [LICENSE](LICENSE).
