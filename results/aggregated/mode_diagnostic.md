# Dark-Mode Threshold Diagnostic (N=90)

> **⚠ Stale diagnostic.** This file was generated from a preliminary 90-image run with threshold L\* < 50. The production evaluation uses **N = 115** (90 val2017 + 25 train2017) and threshold **L\* < 40**, giving 31 dark / 84 light. To regenerate this diagnostic, re-run `evaluation/mode_diagnostic_fig.py` against the full corpus.

## How is mean L\* computed?

Pixel-wise mean over ALL pixels in the full-resolution image, after converting sRGB → XYZ → CIELAB (using `colour.sRGB_to_XYZ` + `colour.XYZ_to_Lab`, D65 illuminant). Each pixel contributes equally regardless of palette frequency. This is **not** frequency-weighted over the quantised palette; it is computed before any palette extraction, in `runner.py` line 113: `image_mean_L = float(img_lab[:, 0].mean())`.

## L\* Histogram (all 90 images)

```
Bin             Count  Bar
-------------------------------------------------------
L* ∈ [ 0, 10)        1  █
L* ∈ [10, 20)        3  ███
L* ∈ [20, 30)        9  █████████
L* ∈ [30, 40)       12  ████████████
L* ∈ [40, 50)       34  ██████████████████████████████████
L* ∈ [50, 60)       18  ██████████████████
L* ∈ [60, 70)       10  ██████████
L* ∈ [70, 80)        3  ███
L* ∈ [80, 90)        0
L* ∈ [90,100)        0
```

## Dark-Mode Counts at Candidate Thresholds

| Threshold | Dark (< L\*) | Light (≥ L\*) | Dark % |
| --------- | ------------ | ------------- | ------ |
| L\* < 40  | 25           | 65            | 27.8%  |
| L\* < 45  | 45           | 45            | 50.0%  |
| L\* < 50  | 59           | 31            | 65.6%  |
| L\* < 55  | 72           | 18            | 80.0%  |

_Current threshold in `runner.py`/`assign_roles`: **L\* < 40** (see `research/dsp/roles.py`). At L\* < 40, this preliminary N=90 run shows 25 dark / 65 light; the full N=115 corpus gives **31 dark / 84 light**._

## All 90 Images Sorted by Mean L\*

| #   | COCO ID           | mean L\* | mode  |
| --- | ----------------- | -------- | ----- |
| 1   | coco_000000255749 | 9.08     | dark  |
| 2   | coco_000000085329 | 14.38    | dark  |
| 3   | coco_000000187249 | 18.21    | dark  |
| 4   | coco_000000418961 | 19.38    | dark  |
| 5   | coco_000000369037 | 20.40    | dark  |
| 6   | coco_000000213035 | 21.23    | dark  |
| 7   | coco_000000119516 | 23.05    | dark  |
| 8   | coco_000000462031 | 25.56    | dark  |
| 9   | coco_000000412286 | 26.60    | dark  |
| 10  | coco_000000004134 | 27.49    | dark  |
| 11  | coco_000000403584 | 27.94    | dark  |
| 12  | coco_000000104666 | 29.50    | dark  |
| 13  | coco_000000380706 | 29.99    | dark  |
| 14  | coco_000000168974 | 30.91    | dark  |
| 15  | coco_000000470779 | 32.02    | dark  |
| 16  | coco_000000307145 | 32.76    | dark  |
| 17  | coco_000000284106 | 34.03    | dark  |
| 18  | coco_000000544811 | 34.16    | dark  |
| 19  | coco_000000438304 | 34.73    | dark  |
| 20  | coco_000000069106 | 34.88    | dark  |
| 21  | coco_000000236690 | 36.62    | dark  |
| 22  | coco_000000101022 | 36.63    | dark  |
| 23  | coco_000000464522 | 37.12    | dark  |
| 24  | coco_000000161128 | 38.27    | dark  |
| 25  | coco_000000002149 | 39.76    | dark  |
| 26  | coco_000000035279 | 40.01    | dark  |
| 27  | coco_000000228214 | 40.11    | dark  |
| 28  | coco_000000393115 | 40.13    | dark  |
| 29  | coco_000000437205 | 40.15    | dark  |
| 30  | coco_000000283520 | 40.73    | dark  |
| 31  | coco_000000537355 | 40.85    | dark  |
| 32  | coco_000000423506 | 40.95    | dark  |
| 33  | coco_000000371472 | 41.19    | dark  |
| 34  | coco_000000166287 | 41.21    | dark  |
| 35  | coco_000000162543 | 41.69    | dark  |
| 36  | coco_000000294163 | 41.76    | dark  |
| 37  | coco_000000186632 | 42.84    | dark  |
| 38  | coco_000000300233 | 42.97    | dark  |
| 39  | coco_000000524742 | 43.21    | dark  |
| 40  | coco_000000233370 | 43.27    | dark  |
| 41  | coco_000000275791 | 43.82    | dark  |
| 42  | coco_000000004395 | 43.92    | dark  |
| 43  | coco_000000028449 | 44.05    | dark  |
| 44  | coco_000000190923 | 44.44    | dark  |
| 45  | coco_000000090062 | 44.87    | dark  |
| 46  | coco_000000184400 | 45.42    | dark  |
| 47  | coco_000000005060 | 45.55    | dark  |
| 48  | coco_000000107226 | 45.71    | dark  |
| 49  | coco_000000542625 | 46.47    | dark  |
| 50  | coco_000000262487 | 47.71    | dark  |
| 51  | coco_000000252216 | 47.76    | dark  |
| 52  | coco_000000002431 | 48.06    | dark  |
| 53  | coco_000000021465 | 48.14    | dark  |
| 54  | coco_000000008277 | 48.24    | dark  |
| 55  | coco_000000244379 | 48.99    | dark  |
| 56  | coco_000000001503 | 49.39    | dark  |
| 57  | coco_000000440171 | 49.56    | dark  |
| 58  | coco_000000002473 | 49.66    | dark  |
| 59  | coco_000000145020 | 50.00    | dark  |
| 60  | coco_000000575372 | 50.07    | light |
| 61  | coco_000000002006 | 50.13    | light |
| 62  | coco_000000281693 | 50.42    | light |
| 63  | coco_000000005037 | 50.55    | light |
| 64  | coco_000000462371 | 50.81    | light |
| 65  | coco_000000553339 | 51.08    | light |
| 66  | coco_000000169169 | 51.14    | light |
| 67  | coco_000000393093 | 51.67    | light |
| 68  | coco_000000002592 | 51.72    | light |
| 69  | coco_000000003501 | 52.31    | light |
| 70  | coco_000000453341 | 54.55    | light |
| 71  | coco_000000001532 | 54.93    | light |
| 72  | coco_000000503841 | 54.96    | light |
| 73  | coco_000000033854 | 55.01    | light |
| 74  | coco_000000117645 | 55.49    | light |
| 75  | coco_000000003661 | 57.35    | light |
| 76  | coco_000000554735 | 58.14    | light |
| 77  | coco_000000542856 | 58.14    | light |
| 78  | coco_000000004765 | 60.07    | light |
| 79  | coco_000000494759 | 61.72    | light |
| 80  | coco_000000396729 | 62.65    | light |
| 81  | coco_000000346232 | 62.88    | light |
| 82  | coco_000000007281 | 63.26    | light |
| 83  | coco_000000002261 | 64.76    | light |
| 84  | coco_000000315187 | 65.43    | light |
| 85  | coco_000000075393 | 67.21    | light |
| 86  | coco_000000512330 | 67.37    | light |
| 87  | coco_000000335427 | 69.61    | light |
| 88  | coco_000000334977 | 71.29    | light |
| 89  | coco_000000288762 | 73.64    | light |
| 90  | coco_000000001761 | 79.82    | light |
