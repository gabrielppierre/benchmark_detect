# det_benchmark_fair_v1

## Protocolo

- dataset: `tower_vision/v2026-04-16`
- split oficial: `official_v1`
- classes: `torre, isoladores`
- classe crítica: `isoladores`
- early stopping: simétrico por `val_map50_95`
- melhor checkpoint: maior `val_map50_95`
- ranking geral: `test_map50_95`
- ranking orientado à classe crítica: `test_AP50_95_isoladores`

## Resultados Consolidados

| Modelo | Seeds | Val mAP50-95 | Val AP50-95 Isoladores | Val Recall Isoladores | Test mAP50-95 | Test AP50-95 Isoladores | Test Recall Isoladores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RT-DETR-R18 | 3/3 | 0.7806 ± 0.0119 | 0.6870 ± 0.0258 | 1.0000 ± 0.0000 | 0.8145 ± 0.0075 | 0.7106 ± 0.0329 | 1.0000 ± 0.0000 |
| YOLO11s | 3/3 | 0.7910 ± 0.0022 | 0.7229 ± 0.0015 | 0.9974 ± 0.0037 | 0.8058 ± 0.0090 | 0.7098 ± 0.0088 | 0.9752 ± 0.0059 |
| YOLOX-s | 3/3 | 0.7423 ± 0.0102 | 0.6616 ± 0.0142 | 1.0000 ± 0.0000 | 0.8000 ± 0.0193 | 0.7054 ± 0.0339 | 1.0000 ± 0.0000 |
| Faster R-CNN ResNet50-FPN v2 | 3/3 | 0.7560 ± 0.0053 | 0.6648 ± 0.0107 | 0.9922 ± 0.0063 | 0.7816 ± 0.0070 | 0.6912 ± 0.0036 | 1.0000 ± 0.0000 |

## Seleção Para Anomalia

- piso de `Recall_isoladores` em validação: `0.850`
- detector selecionado: `YOLO11s`
- critério: maior `val_AP50_95_isoladores` entre os modelos com `val_Recall_isoladores >= 0.850`; desempate por `val_map50_95`