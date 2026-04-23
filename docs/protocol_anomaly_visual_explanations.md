# Anomaly Visual Explanations

## Objetivo

Definir uma trilha complementar de **explicações visuais** para modelos de anomaly detection que não geram mapa espacial nativo comparável.

Na versão atual do projeto, essa trilha existe principalmente para:

- `CutPaste`

## Motivação

O benchmark oficial `anom_benchmark_fair_v1` já mostrou que:

- `CutPaste` foi o melhor método em métricas ROI-level;
- `PatchCore` e `PaDiM` geram `anomaly_map` espacial nativo;
- `CutPaste` não gera `anomaly_map` espacial nativo na implementação atual.

Isso cria uma assimetria prática:

- o melhor modelo no ranking científico não possui, hoje, uma visualização espacial nativa equivalente à dos demais.

Para inspeção qualitativa, auditoria humana e entendimento do comportamento do modelo, vale manter uma trilha separada de explicações visuais.

## O que esta trilha é

Esta trilha é uma trilha de:

- `visual explanation`
- `evidence map`
- `decision explanation`

Ela **não** deve ser tratada como:

- `anomaly_map` nativo;
- segmentação de defeito;
- métrica espacial oficial do benchmark;
- proxy exato da localização da anomalia.

## O que esta trilha não é

Ela não deve ser usada para:

- ranking principal;
- ranking secundário;
- comparação pixel-level entre `CutPaste` e `PatchCore/PaDiM`;
- cálculo oficial de `pixel_auroc`, `pixel_auprc`, `mask_iou` ou `dice`.

## Separação metodológica obrigatória

No repositório, deve haver distinção explícita entre:

- `anomaly_maps/`
  - mapas espaciais nativos de métodos que os produzem, como `PatchCore` e `PaDiM`;
- `visual_explanations/`
  - explicações visuais complementares de métodos sem mapa espacial nativo, como `CutPaste`.

Essas duas trilhas não devem ser mescladas em uma mesma tabela de métrica espacial.

## Método inicial recomendado

Para `CutPaste`, a recomendação inicial é usar:

- `Grad-CAM`

Motivo:

- é uma técnica consolidada de explicação visual;
- é mais honesta como explicação de decisão do que inventar um falso mapa de anomalia;
- preserva a distinção entre "região importante para a decisão" e "localização precisa da anomalia".

## Interpretação correta

Ao analisar um `Grad-CAM` do `CutPaste`, a leitura correta é:

- "esta região contribuiu para o score de anomalia"

e não:

- "esta região é a máscara real do defeito"

Isso é especialmente importante para:

- `crack`
- defeitos pequenos;
- alterações sutis de textura;
- cenários com fundo ou borda muito influentes.

## Saídas canônicas

Raiz canônica proposta:

`runs/anomaly/<dataset>/<version>/<benchmark>/<model>/seed_<seed>/visual_explanations/<split>/`

Arquivos esperados:

- `images/*.png`
- `explanation_index.csv`
- `summary.json`
- `contact_sheet_top_scores.png`
- `contact_sheet_anomalies.png`

Script fino atual:

- `python scripts/render_cutpaste_visual_explanations.py --seeds 42 52 62 --split test`

Campos mínimos de `explanation_index.csv`:

- `roi_id`
- `image_id`
- `split`
- `label`
- `score`
- `prediction`
- `threshold`
- `crop_path`
- `generator_family`
- `anomaly_type`
- `severity`
- `explanation_method`
- `explanation_path`

## Relação com o benchmark oficial

Esta trilha:

- pode ser produzida a partir dos runs oficiais já treinados;
- pode ser usada em revisão qualitativa e auditoria;
- não altera o leaderboard oficial;
- não substitui as métricas ROI-level do benchmark.

## Estado atual

- `PatchCore` e `PaDiM` já têm `anomaly_maps/test/` materializados;
- `CutPaste` já possui `visual_explanations/test/` materializadas nas três seeds `[42, 52, 62]`;
- cada seed do `CutPaste` já contém `157` explicações individuais do split `test`;
- cada seed também contém:
  - `explanation_index.csv`
  - `summary.json`
  - `contact_sheet_top_scores.png`
  - `contact_sheet_anomalies.png`

Exemplos canônicos:

- `runs/anomaly/tower_vision/v2026-04-16/anom_benchmark_fair_v1/cutpaste/seed_42/visual_explanations/test/`
- `runs/anomaly/tower_vision/v2026-04-16/anom_benchmark_fair_v1/cutpaste/seed_52/visual_explanations/test/`
- `runs/anomaly/tower_vision/v2026-04-16/anom_benchmark_fair_v1/cutpaste/seed_62/visual_explanations/test/`

## Achados da revisão espacial

A trilha de `visual_explanations/` já foi confrontada com as máscaras aceitas do split `test`.

Artefatos canônicos:

- `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_review.md`
- `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_comparison_summary.csv`
- `reports/figures/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_review/seed_52_localization_comparison_sheet.png`

Conclusão principal:

- `CutPaste` continua muito forte em **classificação por ROI**;
- mas suas `visual_explanations` ficaram fracas em **localização espacial** da anomalia;
- o padrão observado sugere dependência de contexto, fundo, cabos ou estrutura de suporte em parte dos casos.

Resumo quantitativo da revisão espacial do `CutPaste`:

- `mass_gain_vs_area_mean = 2.388`
- `peak_in_mask_rate = 0.000`
- `dice_top5_mean = 0.058`

Leitura correta:

- o `Grad-CAM` do `CutPaste` é útil para auditoria qualitativa;
- ele não deve ser usado como evidência de localização precisa da falha;
- quando o projeto precisar interpretar espacialmente o defeito, `PatchCore` e `PaDiM` são mais defensáveis do que o `CutPaste`.

## Implicação para o projeto

Com essa separação:

- o projeto preserva rigor metodológico;
- o melhor modelo (`CutPaste`) passa a poder ser inspecionado visualmente;
- evita-se chamar uma explicação visual de "heatmap de anomalia" quando ela não é equivalente aos mapas nativos dos outros métodos;
- e fica explícito que a próxima hipótese forte de pesquisa é reduzir contexto com máscara/segmentação do isolador antes da etapa de anomalia.
