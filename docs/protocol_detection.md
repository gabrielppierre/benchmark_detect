# Detection Protocol

## Objetivo

Comparar detectores supervisionados sob um contrato justo, reprodutível e auditável na versão atual do dataset Tower Vision.

## Unidade de comparação

- treino em `train`;
- early stopping e seleção de checkpoint em `val`;
- avaliação final em `test`;
- classes supervisionadas: `torre` e `isoladores`;
- métrica principal: `mAP50-95`.

## Split oficial v1

- dataset ativo: `tower_vision/v2026-04-16`;
- split oficial: `official_v1`;
- estratégia de agrupamento: bucket temporal de 30 segundos inferido do nome do arquivo;
- ordenação do split: grupos cronológicos contíguos para reduzir vazamento entre frames próximos;
- artefatos do split: `data/splits/tower_vision/v2026-04-16/`.

## Modelos previstos

- YOLO11s
- YOLOX-s
- Faster R-CNN ResNet50-FPN v2
- RT-DETR-R18

## Benchmark justo v1

- config central: `configs/experiment/det_benchmark_fair_v1.yaml`;
- `img_size = 640`;
- `max_epochs = 100`;
- `validate_every = 1`;
- `save_best = true`;
- `save_last = true`;
- `early_stopping = true`;
- `monitor = val_map50_95`;
- `mode = max`;
- `patience = 20`;
- `min_epochs = 25`;
- `num_seeds = 3` por padrão;
- ranking geral: `test_map50_95`;
- ranking orientado à classe crítica: `test_AP50_95_isoladores`;
- decisão para anomalia: maior `val_AP50_95_isoladores` entre os modelos que passam no piso de `val_Recall_isoladores`, com desempate por `val_map50_95`.

## Saídas esperadas

- jobs e logs em `runs/detectors/<dataset>/<version>/<benchmark>/<model>/seed_<seed>/`;
- split materializado do benchmark em `data/interim/<dataset>/<version>/detection_benchmark/<benchmark>/dataset/`;
- métricas por época em `epoch_metrics.csv`;
- resultado por seed em `result.json`;
- consolidação em `reports/tables/<dataset>/<version>/detection_benchmark/<benchmark>/`.
