# Tower Vision

Base de experimentos para visão computacional no projeto Tower Vision, com foco em:

1. detecção supervisionada de isoladores;
2. recorte de ROIs de isoladores a partir de ground truth e predições;
3. detecção não supervisionada de anomalias nas ROIs.

O objetivo desta base é facilitar benchmarks reproduzíveis, com separação clara entre código-fonte, scripts de orquestração, configurações e artefatos de execução.

## Princípios

- `src/` concentra a lógica de negócio;
- `scripts/` apenas conectam configuração, I/O e chamadas para `src/`;
- `data/`, `runs/` e `reports/` recebem artefatos e outputs;
- `configs/` e `params.yaml` definem os experimentos;
- `dvc.yaml` prepara a pipeline para rastrear execução e dependências.

## Estrutura

```text
tower-vision/
├── configs/                # Configurações de dados, detectores, anomaly e experimentos
├── data/                   # Dados brutos, dados preparados, splits e crops
├── docs/                   # Contratos, protocolos, inspeções e material de referência
├── reports/                # Relatórios, tabelas e figuras
├── runs/                   # Artefatos de treino e inferência
├── scripts/                # Entrypoints finos para a pipeline
├── src/towervision/        # Implementação principal
└── tests/                  # Testes unitários do scaffold
```

## Fluxo experimental

1. `prepare_dataset` descobre imagens, carrega anotações e gera manifests limpos.
2. `make_splits` cria splits determinísticos sem vazamento entre subconjuntos.
3. `train_detector` registra um treino placeholder para o detector selecionado.
4. `infer_detector` gera predições placeholder no split de teste.
5. `crop_rois` produz `gt_crops` e `pred_crops` para comparação direta.
6. `train_anomaly` registra um treino placeholder para anomaly detection.
7. `evaluate_pipeline` agrega métricas de detecção e resumos de anomalia.

O scaffold já deixa evidentes os pontos onde entram YOLO, YOLOX, Faster R-CNN, RT-DETR, PatchCore, PaDiM e CutPaste, sem acoplar uma implementação específica agora.

## Comandos principais

```bash
make setup
make setup-benchmark
make prepare-data
make test
make format
make lint
make det-benchmark
make det-benchmark-fair-prepare
make det-benchmark-fair
make anom-benchmark
```

Execução manual dos scripts:

```bash
python scripts/prepare_dataset.py
python scripts/make_splits.py
python scripts/train_detector.py
python scripts/infer_detector.py
python scripts/crop_rois.py --source both
python scripts/train_anomaly.py
python scripts/evaluate_pipeline.py
python scripts/run_detection_benchmark.py
python scripts/run_detection_benchmark.py --execute --models yolo11s --seeds 42
```

## Ambiente Conda

```bash
conda env create -f environment.yml
conda activate tower-vision
```

Para instalar a stack de benchmark via `pip` no ambiente já ativo:

```bash
python -m pip install -e .[dev,benchmark]
```

## Formato de dados

O formato esperado para imagens e anotações está documentado em [docs/dataset.md](docs/dataset.md). Se esse contrato mudar, atualize a documentação antes de expandir a pipeline.

As inspeções do estado atual do dataset são salvas em `docs/inspections/dataset/<timestamp>/dataset_current_state.md` e `reports/tables/dataset/<timestamp>/dataset_summary.csv`, com um atalho estável em `docs/inspections/dataset/latest.md`.

Os datasets brutos devem ficar versionados em `data/raw/<dataset_name>/<dataset_version>/`, com `archive/`, `extracted/` e `manifest.yaml` separados. A versão ativa é apontada por `configs/data/base.yaml` e `params.yaml`.

Materiais de contexto complementar, como markdowns e PDFs de deep research, devem ficar em `docs/references/deepsearch/`. Esse diretório não substitui os protocolos oficiais do repositório, mas serve como base consultiva quando mais contexto for necessário.

Pacotes de anomalias sintéticas controladas devem ficar em `data/synthetic/<dataset>/<version>/<synthetic_pack>/`, sempre separados de `data/raw/`.
O pacote inicial pode ser criado ou refeito com `python scripts/init_synthetic_anomaly_pack.py`.
Os source crops para geração sintética podem ser exportados com `python scripts/prepare_synthetic_source_crops.py`.
Uma pasta única com a shortlist pronta para handoff pode ser gerada com `python scripts/materialize_synthetic_shortlist.py`.
O `records.csv` pode ser sincronizado com os outputs gerados via `python scripts/sync_synthetic_records.py`.
As máscaras anotadas no Roboflow podem ser importadas com `python scripts/import_roboflow_masks.py --export-root dataset_segmentation_roboflow`; nesse passo, o split exportado pelo Roboflow não é usado, e `val/test` são recuperados do `records.csv`.
As overlays de revisão das máscaras podem ser geradas com `python scripts/render_synthetic_mask_overlays.py`.
Esse comando também gera uma prancha única em `.../mask_overlays/contact_sheet.png`.
Após a curadoria, os registros aceitos podem ser promovidos com `python scripts/accept_synthetic_records.py`.
Estado atual do pack `anomaly_controlled_v1`: `20` imagens sintéticas aceitas, máscaras internalizadas em `masks/val` e `masks/test`, e prancha de revisão em `reports/figures/tower_vision/v2026-04-16/anomaly_controlled_v1/mask_overlays/contact_sheet.png`.

## Benchmark Justo de Detecção

O benchmark harmonizado da versão atual do dataset é definido em `configs/experiment/det_benchmark_fair_v1.yaml` e consome o split congelado `official_v1` sem recriá-lo.

- preparação: `python scripts/run_detection_benchmark.py`
- execução: `python scripts/run_detection_benchmark.py --execute`
- logs por seed/modelo: `runs/detectors/<dataset>/<version>/<benchmark>/<model>/seed_<seed>/train.log`
- métricas por época: `.../epoch_metrics.csv`
- resultado consolidado: `reports/tables/<dataset>/<version>/detection_benchmark/<benchmark>/`
- relatório markdown canônico: `reports/tables/<dataset>/<version>/detection_benchmark/<benchmark>/benchmark_report.md`

Para acompanhar um treino em execução:

```bash
tail -f runs/detectors/tower_vision/v2026-04-16/det_benchmark_fair_v1/yolo11s/seed_42/train.log
```

Estado atual dos backends executáveis: `YOLO11s`, `YOLOX-s`, `Faster R-CNN ResNet50-FPN v2` e `RT-DETR-R18`, todos já executados no benchmark justo v1.

## Benchmark de Anomalia

O protocolo oficial atual do benchmark não supervisionado é `anom_benchmark_fair_v1`.

- config central: `configs/experiment/anom_benchmark_fair_v1.yaml`
- protocolo específico: `docs/protocol_anomaly_benchmark_fair_v1.md`
- protocolo complementar de explicabilidade: `docs/protocol_anomaly_visual_explanations.md`
- script fino: `python scripts/run_anomaly_benchmark.py`
- preparação: `python scripts/run_anomaly_benchmark.py`
- execução: `python scripts/run_anomaly_benchmark.py --execute --models patchcore --seeds 42`
- heatmaps pós-benchmark: `python scripts/render_anomaly_heatmaps.py --models patchcore padim --seeds 42 52 62 --split test`
- visual explanations do CutPaste: `python scripts/render_cutpaste_visual_explanations.py --seeds 42 52 62 --split test`
- logs por seed/modelo: `runs/anomaly/<dataset>/<version>/<benchmark>/<model>/seed_<seed>/train.log`
- saída canônica: `reports/tables/<dataset>/<version>/anomaly_benchmark/<benchmark>/`

Estado atual da implementação:

- o runner do benchmark já materializa `train/val/test`, agenda jobs e consolida relatórios;
- `PatchCore` e `PaDiM` já rodam via `anomalib`;
- `CutPaste` já roda via implementação local do repositório, com checkpoint `best/last` e early stopping em `val_roi_auroc`;
- o benchmark oficial `anom_benchmark_fair_v1` já foi concluído nas três seeds `[42, 52, 62]` para as três famílias.

Resumo consolidado atual:

- `CutPaste`: `test_roi_auroc=0.9880`, `test_roi_auprc=0.9178`
- `PatchCore`: `test_roi_auroc=0.9517`, `test_roi_auprc=0.6401`
- `PaDiM`: `test_roi_auroc=0.9494`, `test_roi_auprc=0.5336`

Leitura atual:

- vencedor científico do v1 em `gt_crops`: `CutPaste`
- comparador relevante orientado a recall: `PaDiM`
- revisão espacial complementar: `CutPaste` foi o pior dos três em alinhamento espacial com a máscara da anomalia
- próximo passo metodológico: comparar `gt_crops` vs `gt_masked_crops`, antes da trilha `pred_crops`

Mapas de calor:

- `PatchCore` e `PaDiM` já suportam materialização de heatmaps de teste em `runs/anomaly/<dataset>/<version>/<benchmark>/<model>/seed_<seed>/anomaly_maps/test/`;
- cada run materializa `157` heatmaps individuais do split `test`, `heatmap_index.csv`, `summary.json`, `contact_sheet_top_scores.png` e `contact_sheet_anomalies.png`;
- `CutPaste` não gera heatmap espacial confiável na implementação atual; por isso o script registra o run como `supported=false` em vez de inventar uma visualização artificial.

Explicabilidade complementar:

- para `CutPaste`, a trilha correta é `visual_explanations/`, separada de `anomaly_maps/`;
- essa trilha já foi materializada com `Grad-CAM` em `runs/anomaly/<dataset>/<version>/<benchmark>/cutpaste/seed_<seed>/visual_explanations/test/`;
- ela não deve ser tratada como mapa de anomalia nativo nem entrar no ranking pixel-level;
- a revisão espacial consolidada está em `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_review.md`.

Relatório canônico atual:

- `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/benchmark_report.md`
- interpretação técnica curada: `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/benchmark_interpretation.md`

## Benchmark inicial

- detecção: baseline com configuração de detector parametrizada e métrica `f1@0.5iou`;
- anomaly detection: baseline com treino em `gt_crops` e comparação de inferência em `gt_crops` versus `pred_crops`;
- rastreamento do baseline inicial: `dvc.yaml`, `params.yaml`, `reports/benchmark_v1.md`.
