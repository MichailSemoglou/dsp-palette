# Summary for Paper — DSP Palette Extraction

_Generated automatically from N=115 COCO corpus (90 val2017 + 25 train2017 check)._

## 1. Statistical Conventions

- **All p-values are two-tailed** Wilcoxon signed-rank tests (paired, one-sample on DSP − baseline differences).
- **Effect size:** Cliff's δ (DSP − baseline direction; positive = DSP tends higher).
- **DSP vs k-Means RGB, WCAG AA Coverage:** two-tailed p=0.013 (significant at α=0.05 but does not survive Bonferroni correction, α_corrected ≈ 0.004). Marked \* in figures but not \*\* in tables.
- **Bonferroni correction** (12 baseline × metric comparisons, α=0.05/12 ≈ 0.004): all min ΔE₂₀₀₀ comparisons, all reconstruction comparisons, the WCAG AA/k-Means Lab comparison (p=1.8e-05), and the WCAG AA/Median Cut comparison (p=0.0014) survive correction. The WCAG AA/k-Means RGB comparison (p=0.013) is nominally significant but does not survive Bonferroni correction. Harmony alignment comparisons remain non-significant under correction.
- **β/α invariance:** Selection is empirically invariant to β/α ∈ [0.25, 4] (spread < 1 ΔE₂₀₀₀ on N=30 images). All results use α=β=1.0. See `research/dsp/selector.py` docstring for structural explanation.

## 2. Corpus Composition

- **N = 115** images
- Source: COCO val2017 (`http://cocodataset.org`), CC BY 4.0
- Sampling: 90 COCO val2017 images (seed 2026) drawn from the full 5,000-image val split, plus 25 COCO train2017 images (ranking-check set)
- Subset split: 15 dev (`dev=true` in manifest), 75 test (val2017), 25 check (train2017)
- `mode=auto` → **dark-mode** (mean L\* < 40): **31/115** images
- `mode=auto` → **light-mode** (mean L\* ≥ 40): **84/115** images
- Images triggering `wcag_distinctness_compromised=True`: **0/115**

## 3. Per-Method Aggregate Stats (all four metrics)

### Min Delta E\_{2000} uparrow

| Method      | Mean    | Std    | Median  |
| ----------- | ------- | ------ | ------- |
| DSP         | 24.9207 | 5.6423 | 25.0071 |
| k-Means Lab | 14.0673 | 4.8964 | 13.5151 |
| k-Means RGB | 12.3826 | 3.7908 | 12.5535 |
| Median Cut  | 7.1824  | 3.8937 | 6.7899  |

### WCAG AA Coverage uparrow

| Method      | Mean   | Std    | Median |
| ----------- | ------ | ------ | ------ |
| DSP         | 0.3307 | 0.0900 | 0.3000 |
| k-Means Lab | 0.2613 | 0.1077 | 0.3000 |
| k-Means RGB | 0.3000 | 0.1151 | 0.3000 |
| Median Cut  | 0.2667 | 0.1545 | 0.3000 |

### Recon. Delta E\_{2000} downarrow

| Method      | Mean   | Std    | Median |
| ----------- | ------ | ------ | ------ |
| DSP         | 9.7828 | 3.4282 | 9.7213 |
| k-Means Lab | 6.4801 | 2.0899 | 6.6790 |
| k-Means RGB | 6.7512 | 2.2930 | 7.0050 |
| Median Cut  | 7.6157 | 2.4838 | 7.5074 |

### Harmony Alignment uparrow

| Method      | Mean   | Std    | Median |
| ----------- | ------ | ------ | ------ |
| DSP         | 0.4620 | 0.1533 | 0.4818 |
| k-Means Lab | 0.4155 | 0.1683 | 0.4289 |
| k-Means RGB | 0.4264 | 0.1810 | 0.4782 |
| Median Cut  | 0.4213 | 0.1812 | 0.4430 |

## 4. Wilcoxon Signed-Rank Tests (DSP vs Baseline)

All tests two-tailed. Cliff's δ = (DSP − baseline) direction; positive means DSP tends higher.

| Baseline    | Metric                           | W-stat | p-value  | Cliff's δ | Magnitude  | Sig    |
| ----------- | -------------------------------- | ------ | -------- | --------- | ---------- | ------ |
| k-Means Lab | Harmony Alignment uparrow        | 1050.0 | 4.77e-02 | +0.1524   | small      | \*     |
| k-Means RGB | Harmony Alignment uparrow        | 1281.0 | 4.47e-01 | +0.0535   | negligible | ns     |
| Median Cut  | Harmony Alignment uparrow        | 1116.0 | 1.03e-01 | +0.0898   | negligible | ns     |
| k-Means Lab | Min Delta E\_{2000} uparrow      | 3.0    | 0.00e+00 | +0.8457   | large      | \*\*\* |
| k-Means RGB | Min Delta E\_{2000} uparrow      | 1.0    | 0.00e+00 | +0.9332   | large      | \*\*\* |
| Median Cut  | Min Delta E\_{2000} uparrow      | 0.0    | 0.00e+00 | +0.9886   | large      | \*\*\* |
| k-Means Lab | Recon. Delta E\_{2000} downarrow | 0.0    | 0.00e+00 | +0.6050   | large      | \*\*\* |
| k-Means RGB | Recon. Delta E\_{2000} downarrow | 12.0   | 0.00e+00 | +0.5516   | large      | \*\*\* |
| Median Cut  | Recon. Delta E\_{2000} downarrow | 260.0  | 0.00e+00 | +0.4265   | medium     | \*\*\* |
| k-Means Lab | WCAG AA Coverage uparrow         | 249.0  | 1.80e-05 | +0.3797   | medium     | \*\*\* |
| k-Means RGB | WCAG AA Coverage uparrow         | 401.0  | 1.30e-02 | +0.1591   | small      | \*     |
| Median Cut  | WCAG AA Coverage uparrow         | 408.5  | 1.41e-03 | +0.2633   | small      | \*\*   |

## 5. Per-Image Win Counts (DSP vs Baselines on min ΔE2000)

- DSP > **k-Means Lab** on min ΔE₂₀₀₀: **73/75**
- DSP > **k-Means RGB** on min ΔE₂₀₀₀: **74/75**
- DSP > **Median Cut** on min ΔE₂₀₀₀: **75/75**

- DSP > k-Means Lab on WCAG AA Coverage uparrow: **40/75**
- DSP > k-Means RGB on WCAG AA Coverage uparrow: **29/75**
- DSP > Median Cut on WCAG AA Coverage uparrow: **38/75**
- DSP < k-Means Lab on Recon. Delta E\_{2000} downarrow: **0/75**
- DSP < k-Means RGB on Recon. Delta E\_{2000} downarrow: **1/75**
- DSP < Median Cut on Recon. Delta E\_{2000} downarrow: **9/75**
- DSP > k-Means Lab on Harmony Alignment uparrow: **46/75**
- DSP > k-Means RGB on Harmony Alignment uparrow: **40/75**
- DSP > Median Cut on Harmony Alignment uparrow: **40/75**

## 6. Worst-Five DSP Images (lowest min ΔE2000)

| Rank | Image ID          | min ΔE2000 | AA Cov | Recon ΔE | Harmony |
| ---- | ----------------- | ---------- | ------ | -------- | ------- |
| 1    | coco_000000117645 | 11.284     | 0.100  | 2.921    | 0.440   |
| 2    | coco_000000162543 | 11.444     | 0.200  | 12.858   | 0.683   |
| 3    | coco_000000396729 | 15.867     | 0.200  | 3.642    | 0.105   |
| 4    | coco_000000553339 | 17.067     | 0.300  | 6.562    | 0.509   |
| 5    | coco_000000236690 | 17.135     | 0.400  | 6.807    | 0.159   |

## 7. Figure 2 Selected Images

- **Colourful image**: `coco_000000288762` (mean L\*=73.6, min ΔE=37.77, recon=13.02)
- **High-variance image**: `coco_000000021465` (mean L\*=48.1, min ΔE=31.80, recon=17.12)
- **Low-chroma image**: `coco_000000117645` (mean L\*=55.5, min ΔE=11.28, recon=2.92)
- **Dark image (mode=auto)**: `coco_000000438304` (mean L\*=34.7, min ΔE=33.44, recon=10.14)

## 8. Metric Definitions

**ΔE₂₀₀₀ (CIE 2000 colour difference):** Computed using the CIEDE2000 formula with standard weighting factors k*L=k_C=k_H=1.0. Reference: CIE 142-2001, *Improvement to Industrial Colour-Difference Evaluation*, Commission Internationale de l'Éclairage, Vienna, 2001. Implementation: `colour.delta_E(..., method='CIE 2000')` from colour-science 0.4.4.
**WCAG AA contrast ratio:** Relative luminance Y computed as per W3C WCAG 2.1 §1.4.3 (linearise sRGB channel c: c/12.92 if c≤0.04045, else ((c+0.055)/1.055)^2.4; then Y = 0.2126R + 0.7152G + 0.0722B). Contrast ratio = (Y_lighter + 0.05) / (Y_darker + 0.05). AA pass threshold: ratio ≥ 4.5 (normal text). WCAG AA Coverage = fraction of ordered colour pairs in the palette that meet this threshold. Reference: W3C, *Web Content Accessibility Guidelines (WCAG) 2.1*, https://www.w3.org/TR/WCAG21/, 2018.
**Harmony alignment:** align(θ) = max*{c ∈ {30°,60°,90°,120°,180°}} exp(−((θ−c)/σ)²), σ=15°. θ is the hue-angle span of the palette in CIELAB (arctan2(b*,a*)), computed as the largest gap-subtracted arc. Score ∈ [0, 1]; 1.0 = perfectly aligned with a canonical harmony template.
**Reconstruction error (ΔE₂₀₀₀):** For each pixel in the image, find the nearest palette colour in CIELAB Euclidean distance, then compute ΔE₂₀₀₀ between the pixel and that palette colour. Reconstruction error = mean over all pixels. Lower is better (palette captures the image colours faithfully).
**Mean L\* (mode threshold):** Pixel-wise mean of the L\* channel after converting the full image from sRGB → XYZ → CIELAB (D65, `colour.sRGB_to_XYZ` + `colour.XYZ_to_Lab`). Computed before palette extraction. Images with mean L\* < 40 are assigned mode=dark, else mode=light.

## 9. Runtime (Wall-Clock, Single Core)

Measured on Apple M3 Max (36 GB RAM), Python 3.12.3, single-threaded. 5 repeats per image × 4 representative Fig 2 images (sizes 640×425, 500×281, 640×480, 464×640). Values are mean wall-clock time per image (ms).

| Method      | Mean time / image |
| ----------- | ----------------- |
| DSP         | 322.9 ms          |
| k-Means Lab | 314.7 ms          |
| k-Means RGB | 69.5 ms           |
| Median Cut  | 173.8 ms          |

DSP overhead vs k-Means Lab (~2.6%) comes from the WCAG distinctness constraint loop and role-assignment heuristic, not from the core clustering step (both use CIELAB k-means internally). Runtime scales with image pixel count; all methods are CPU-bound on the k-means step.
