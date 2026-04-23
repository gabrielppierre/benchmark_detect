# Spatial Localization Review

## Scope
- Benchmark: `anom_benchmark_fair_v1`
- Dataset: `tower_vision/v2026-04-16`
- Split analyzed: `test`
- Positive ROIs analyzed: `10` synthetic anomalies x `3` seeds = `30` maps per model
- Compared artifacts: native `anomaly_maps` for `PatchCore` and `PaDiM`, and `visual_explanations` (Grad-CAM) for `CutPaste`

## Metrics used
- `mass_in_mask`: fraction of total map energy inside the anomaly mask
- `mass_gain_vs_area`: concentration gain relative to mask area; `>1` means stronger-than-random concentration
- `peak_in_mask_rate`: fraction of cases where the strongest activation pixel falls inside the mask
- `dice_top5`: Dice overlap between the anomaly mask and the top 5% highest-activation pixels
- `mean_lift_in_vs_out`: mean activation inside the mask divided by mean activation outside

## Aggregate results
- `padim`: mass_gain `12.692`, peak_in_mask `0.533`, dice_top5 `0.214`, lift `13.951
- `patchcore`: mass_gain `3.253`, peak_in_mask `0.433`, dice_top5 `0.137`, lift `3.310
- `cutpaste`: mass_gain `2.388`, peak_in_mask `0.000`, dice_top5 `0.058`, lift `2.411

## Interpretation
- By spatial alignment, the best model in this comparison is `padim` and the weakest is `cutpaste`.
- `CutPaste` can win ROI-level classification while still placing much of its evidence outside the defect region. That pattern is compatible with a model relying on coarse or contextual cues.
- `PatchCore` and `PaDiM` are not clean localizers either; both leak into background, cables, and support structure. The difference is that they still produce native spatial maps and can be evaluated directly against the masks.
- The seed-52 side-by-side sheet is the most useful visual artifact for manual verification of this conclusion.

## Practical implication
- Keep `CutPaste` as the current ROI-level winner of the anomaly benchmark.
- Do not interpret `CutPaste` Grad-CAM as precise defect localization.
- For spatial interpretability and debugging, `PatchCore` and `PaDiM` remain more defensible than `CutPaste`.

## Outputs
- Detailed per-map metrics: `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_comparison_test.csv`
- Summary by model: `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_comparison_summary.csv`
- Summary by anomaly type: `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_comparison_by_anomaly_type.csv`
- Side-by-side sheet (seed 52): `reports/figures/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_review/seed_52_localization_comparison_sheet.png`