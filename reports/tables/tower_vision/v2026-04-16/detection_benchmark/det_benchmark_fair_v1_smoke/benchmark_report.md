# det_benchmark_fair_v1_smoke

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
| YOLO11s | 1/1 | 0.3933 ± 0.0000 | 0.1723 ± 0.0000 | 0.7527 ± 0.0000 | 0.5220 ± 0.0000 | 0.2536 ± 0.0000 | 0.9014 ± 0.0000 |

## Seleção Para Anomalia

- piso de `Recall_isoladores` em validação: `0.850`
- nenhum modelo completado atingiu o piso configurado