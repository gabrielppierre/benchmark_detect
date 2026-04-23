# anom_benchmark_fair_v1

## Protocolo

- dataset: `tower_vision/v2026-04-16`
- trilha oficial: `gt_crops`
- pack sintético: `anomaly_controlled_v1`
- treino: apenas ROIs normais de `isoladores`
- calibração de threshold: somente em `val`
- ranking principal: `test_roi_auroc`
- ranking secundário: `test_roi_auprc`
- piso operacional de recall em validação: `0.90`

## Resultados Consolidados

| Modelo | Seeds | Val ROI AUROC | Val ROI AUPRC | Val F1 | Val Recall | Test ROI AUROC | Test ROI AUPRC | Test F1 | Test Recall | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CutPaste | 3/3 | 0.9982 ± 0.0020 | 0.9845 ± 0.0160 | 0.8486 ± 0.0288 | 0.8333 ± 0.1247 | 0.9880 ± 0.0075 | 0.9178 ± 0.0343 | 0.8083 ± 0.0478 | 0.8667 ± 0.0943 | 3.1417 ± 0.3210 |
| PatchCore | 3/3 | 0.9189 ± 0.0203 | 0.6771 ± 0.0048 | 0.4277 ± 0.0516 | 0.9000 ± 0.0000 | 0.9517 ± 0.0086 | 0.6401 ± 0.0087 | 0.5072 ± 0.0405 | 0.6667 ± 0.0471 | 0.3169 ± 0.1080 |
| PaDiM | 3/3 | 0.9182 ± 0.0355 | 0.4012 ± 0.1510 | 0.5307 ± 0.1149 | 0.9667 ± 0.0471 | 0.9494 ± 0.0069 | 0.5336 ± 0.0817 | 0.5420 ± 0.0330 | 0.6667 ± 0.1700 | 0.4755 ± 0.0346 |

## Breakdown por Generator Family

| Modelo | Grupo | ROI AUROC | ROI AUPRC | F1 | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| CutPaste | chatgpt | 0.9832 ± 0.0145 | 0.8837 ± 0.0458 | 0.7294 ± 0.0998 | 0.8667 ± 0.0943 |
| CutPaste | gemini | 0.9927 ± 0.0046 | 0.8875 ± 0.0396 | 0.7294 ± 0.0998 | 0.8667 ± 0.0943 |
| PaDiM | chatgpt | 0.9610 ± 0.0051 | 0.3906 ± 0.0929 | 0.4055 ± 0.0521 | 0.6667 ± 0.2494 |
| PaDiM | gemini | 0.9379 ± 0.0089 | 0.4433 ± 0.1013 | 0.4237 ± 0.0331 | 0.6667 ± 0.0943 |
| PatchCore | chatgpt | 0.9655 ± 0.0074 | 0.5037 ± 0.0200 | 0.4013 ± 0.0560 | 0.7333 ± 0.0943 |
| PatchCore | gemini | 0.9379 ± 0.0196 | 0.5595 ± 0.0125 | 0.3414 ± 0.0248 | 0.6000 ± 0.0000 |

## Breakdown por Anomaly Type

| Modelo | Grupo | ROI AUROC | ROI AUPRC | F1 | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| CutPaste | burn mark | 0.9841 ± 0.0128 | 0.6944 ± 0.0786 | 0.4545 ± 0.0643 | 0.6667 ± 0.2357 |
| CutPaste | crack | 0.9580 ± 0.0362 | 0.6282 ± 0.0902 | 0.4545 ± 0.0643 | 0.6667 ± 0.2357 |
| CutPaste | localized surface damage | 0.9989 ± 0.0016 | 0.9444 ± 0.0786 | 0.6545 ± 0.2057 | 1.0000 ± 0.0000 |
| CutPaste | partial chipping | 0.9989 ± 0.0016 | 0.9444 ± 0.0786 | 0.6545 ± 0.2057 | 1.0000 ± 0.0000 |
| CutPaste | severe contamination | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.6545 ± 0.2057 | 1.0000 ± 0.0000 |
| PaDiM | burn mark | 0.9773 ± 0.0058 | 0.5442 ± 0.0912 | 0.2407 ± 0.0131 | 0.6667 ± 0.2357 |
| PaDiM | crack | 0.9274 ± 0.0016 | 0.1567 ± 0.0417 | 0.2407 ± 0.0131 | 0.6667 ± 0.2357 |
| PaDiM | localized surface damage | 0.9116 ± 0.0268 | 0.2756 ± 0.2003 | 0.2019 ± 0.0498 | 0.5000 ± 0.0000 |
| PaDiM | partial chipping | 0.9626 ± 0.0100 | 0.2710 ± 0.0751 | 0.2407 ± 0.0131 | 0.6667 ± 0.2357 |
| PaDiM | severe contamination | 0.9683 ± 0.0163 | 0.2825 ± 0.0820 | 0.3000 ± 0.0707 | 0.8333 ± 0.2357 |
| PatchCore | burn mark | 0.9966 ± 0.0000 | 0.8333 ± 0.0000 | 0.2952 ± 0.0280 | 1.0000 ± 0.0000 |
| PatchCore | crack | 0.9150 ± 0.0083 | 0.1103 ± 0.0128 | 0.1082 ± 0.0782 | 0.3333 ± 0.2357 |
| PatchCore | localized surface damage | 0.8605 ± 0.0488 | 0.0769 ± 0.0163 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| PatchCore | partial chipping | 0.9932 ± 0.0000 | 0.5833 ± 0.0000 | 0.2952 ± 0.0280 | 1.0000 ± 0.0000 |
| PatchCore | severe contamination | 0.9932 ± 0.0000 | 0.5833 ± 0.0000 | 0.2952 ± 0.0280 | 1.0000 ± 0.0000 |

## Breakdown por Severity

| Modelo | Grupo | ROI AUROC | ROI AUPRC | F1 | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| CutPaste | moderate | 0.9880 ± 0.0075 | 0.9178 ± 0.0343 | 0.8083 ± 0.0478 | 0.8667 ± 0.0943 |
| PaDiM | moderate | 0.9494 ± 0.0069 | 0.5336 ± 0.0817 | 0.5420 ± 0.0330 | 0.6667 ± 0.1700 |
| PatchCore | moderate | 0.9517 ± 0.0086 | 0.6401 ± 0.0087 | 0.5072 ± 0.0405 | 0.6667 ± 0.0471 |