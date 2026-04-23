# Detection Benchmark Fair v1

## Objetivo

Implementar um benchmark justo, reproduzível e defensável para comparar:

- `YOLO11s`
- `YOLOX-s`
- `Faster R-CNN ResNet50-FPN v2`
- `RT-DETR-R18`

no split congelado `official_v1` do dataset `tower_vision/v2026-04-16`.

## Regras metodológicas

- não recriar o split;
- não usar `test` para tuning;
- usar pesos COCO-pretrained para todos;
- usar `img_size = 640` na trilha principal;
- usar augmentations leves e comuns;
- validar a cada época;
- usar `val_map50_95` como:
  - monitor de early stopping;
  - critério de melhor checkpoint;
- nunca ranquear famílias por loss;
- nunca deixar o framework escolher implicitamente um critério diferente de melhor checkpoint.

## Configuração padrão

- `max_epochs = 100`
- `validate_every = 1`
- `save_best = true`
- `save_last = true`
- `early_stopping = true`
- `monitor = val_map50_95`
- `mode = max`
- `patience = 20`
- `min_epochs = 25`
- `num_seeds = 3`

Config central: `configs/experiment/det_benchmark_fair_v1.yaml`

## Ranking

- ranking geral: `test_map50_95`
- ranking orientado à classe crítica: `test_AP50_95_isoladores`

## Decisão para anomalia

- aplicar um piso em `val_Recall_isoladores`;
- entre os modelos que passam nesse piso, escolher maior `val_AP50_95_isoladores`;
- usar `val_map50_95` como desempate.

## Artefatos

- dataset materializado: `data/interim/tower_vision/v2026-04-16/detection_benchmark/det_benchmark_fair_v1/dataset/`
- jobs: `runs/detectors/tower_vision/v2026-04-16/det_benchmark_fair_v1/*/seed_*/job.json`
- log de treino: `.../train.log`
- métricas por época: `.../epoch_metrics.csv`
- resultado por seed: `.../result.json`
- consolidação: `reports/tables/tower_vision/v2026-04-16/detection_benchmark/det_benchmark_fair_v1/`

## Como reproduzir

Preparar o benchmark:

```bash
python scripts/run_detection_benchmark.py
```

Executar:

```bash
python scripts/run_detection_benchmark.py --execute
```

Executar subconjunto:

```bash
python scripts/run_detection_benchmark.py --execute --models yolo11s --seeds 42
```

Monitorar:

```bash
tail -f runs/detectors/tower_vision/v2026-04-16/det_benchmark_fair_v1/yolo11s/seed_42/train.log
```

## Estado atual de implementação

- split congelado reutilizado e materializado em views COCO e Ultralytics;
- runner unificado, job specs e consolidação implementados;
- backend Ultralytics para `YOLO11s` validado ponta a ponta em smoke test;
- backend TorchVision para `Faster R-CNN ResNet50-FPN v2` implementado com treino, avaliação COCO, checkpoint `best`/`last` e early stopping por `val_map50_95`;
- backend `YOLOX-s` implementado com treino oficial, checkpoint por `val_map50_95` e consolidação no schema comum do benchmark;
- backend `RT-DETR-R18` implementado com treino via Transformers, checkpoint por `val_map50_95` e consolidação no schema comum do benchmark;
- benchmark completo executado nas 4 famílias com `3` seeds por modelo.

## Resultado do benchmark v1

- vencedor geral de detecção no protocolo fechado: `RT-DETR-R18`, pelo maior `test_map50_95`;
- detector operacional inicial para alimentar a etapa de anomalia em `isoladores`: `YOLO11s`, pelo maior `val_AP50_95_isoladores` entre os modelos que passaram no piso de `val_Recall_isoladores`;
- a escolha operacional não usa `test` para selecionar detector da pipeline;
- o relatório consolidado oficial está em `reports/tables/tower_vision/v2026-04-16/detection_benchmark/det_benchmark_fair_v1/benchmark_report.md`.

## Interpretação correta do resultado

- o benchmark responde qual família teve melhor desempenho global no protocolo de detecção;
- a decisão de sistema responde qual detector melhor atende a etapa seguinte da pipeline;
- nesta versão, como a etapa seguinte prioriza ROIs de `isoladores`, a decisão operacional foi orientada por validação da classe crítica, não pelo ranking global de teste;
- portanto, `RT-DETR-R18` é o vencedor científico do benchmark geral, enquanto `YOLO11s` é o candidato operacional inicial para geração de ROIs de `isoladores`.

## Sistema híbrido

- o repositório não proíbe usar detectores diferentes para regiões ou classes diferentes;
- por exemplo, um detector pode ser usado para `torre` e outro para `isoladores`, desde que a composição seja documentada como sistema híbrido;
- isso é especialmente plausível se a etapa de anomalia evoluir para cobrir falhas estruturais mais amplas, como treliça invertida, ferrugem, empenamento ou outros defeitos em regiões da torre;
- porém, um sistema híbrido não deve ser confundido com o benchmark de detector único;
- se a equipe adotar esse caminho, o sistema híbrido deve ganhar protocolo próprio, regra explícita de fusão das saídas e avaliação dedicada.

## Limitações conhecidas

- a localização canônica da consolidação é `reports/tables/tower_vision/v2026-04-16/detection_benchmark/det_benchmark_fair_v1/`;
- para os adapters customizados, `precision` e `recall` são métricas simples de matching em `IoU=0.5` com `score >= 0.001`; AP continua vindo do COCOeval;
- o split continua heurístico por tempo de captura, então ainda não há garantia absoluta de independência sem metadados fortes como `tower_id` ou `flight_id`.
