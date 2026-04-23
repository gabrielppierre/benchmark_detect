# Anomaly Benchmark Fair v1

## Objetivo

Implementar um benchmark justo, reproduzível e defensável para comparar:

- `PatchCore`
- `PaDiM`
- `CutPaste`

na tarefa de anomaly detection não supervisionada sobre ROIs de `isoladores`, usando o dataset `tower_vision/v2026-04-16` e o pack sintético aceito `anomaly_controlled_v1`.

## Escopo oficial desta versão

- escopo oficial v1: **ROI anomaly detection em `gt_crops`**;
- treino: somente ROIs normais de `isoladores`;
- calibração: split `val`;
- teste final: split `test`;
- anomalias: somente imagens sintéticas aceitas em `data/synthetic/tower_vision/v2026-04-16/anomaly_controlled_v1/`;
- a trilha com `pred_crops` permanece como extensão futura e **não faz parte do leaderboard oficial v1**.

Config central: `configs/experiment/anom_benchmark_fair_v1.yaml`

## Regras metodológicas

- não usar o conjunto `test` para calibrar threshold, escolher época ou decidir modelo;
- não misturar anomalias sintéticas no treino;
- treinar somente com ROIs normais de `isoladores` do split `train`;
- usar a mesma resolução de entrada e o mesmo extrator base na trilha principal;
- manter a comparação principal em métricas **threshold-free**;
- usar métricas **thresholded** apenas para decisão operacional e análise complementar;
- manter a trilha espacial separada da trilha principal, porque nem todo método gera mapa de anomalia comparável.

## Composição do benchmark

### Treino

- conjunto: `gt_crops` normais de `isoladores` do split `train`;
- rótulo esperado: somente normalidade;
- uso de anomalias sintéticas no treino: **proibido** nesta versão.

### Validação

- negativos: `gt_crops` normais de `isoladores` do split `val`;
- positivos: registros do pack sintético com `accepted_for_benchmark=true` e `source_split=val`;
- uso: seleção de checkpoint para métodos iterativos e calibração de threshold operacional.

### Teste

- negativos: `gt_crops` normais de `isoladores` do split `test`;
- positivos: registros do pack sintético com `accepted_for_benchmark=true` e `source_split=test`;
- uso: ranking final e relatório oficial.

## Harmonização entre modelos

### Pré-processamento comum

- `input_size = 256`
- normalização `imagenet`
- `feature_extractor = resnet18` na trilha principal

### Sementes

- `num_seeds = 3`
- `seeds = [42, 52, 62]`

Mesmo para métodos quase determinísticos, as mesmas seeds devem ser registradas e reportadas para manter simetria com o benchmark de detecção e permitir capturar qualquer variação residual de inicialização, dataloader ou subsampling.

### Métodos iterativos vs não iterativos

- `PatchCore` e `PaDiM`: tratados como métodos de ajuste único (`fit_once`), sem early stopping;
- `CutPaste`: tratado como método iterativo, com seleção de checkpoint em validação.

Para métodos iterativos, a regra oficial é:

- `max_epochs = 100`
- `validate_every = 1`
- `save_best = true`
- `save_last = true`
- `early_stopping = true`
- `monitor = val_roi_auroc`
- `mode = max`
- `patience = 20`
- `min_epochs = 25`

## Métricas oficiais

## Ranking principal

- métrica principal: `test_roi_auroc`
- métrica secundária: `test_roi_auprc`

Justificativa:

- `AUROC` é threshold-free e adequada para comparar a separação entre normal e anômalo;
- `AUPRC` entra como métrica secundária porque o problema é assimétrico e a classe anômala é minoritária no conjunto total de ROIs normais do split.

## Métricas thresholded

Estas métricas não definem o ranking principal, mas devem ser reportadas:

- `f1`
- `precision`
- `recall`
- `accuracy`
- `tp`
- `tn`
- `fp`
- `fn`

### Seleção do threshold operacional

O threshold operacional deve ser definido **somente em validação**.

Regra oficial:

- objetivo primário: maior `val_f1`
- restrição: `val_recall >= 0.90`
- desempate: maior `val_precision`
- fallback: se nenhum threshold atingir o piso de recall, usar o threshold de maior `val_f1`

Isso separa:

- **ranking científico**: baseado em métricas threshold-free no `test`;
- **decisão operacional**: baseada em threshold calibrado no `val`.

## Métricas estratificadas

Além do score global, o benchmark deve reportar breakdowns por:

- `generator_family`
- `anomaly_type`
- `severity`

Esses cortes não redefinem o ranking principal, mas são obrigatórios para interpretar:

- robustez a diferentes geradores sintéticos;
- sensibilidade por tipo de anomalia;
- degradação por severidade.

## Métricas espaciais

Como o pack v1 possui máscaras, o benchmark pode reportar métricas espaciais **quando o método gerar mapa de anomalia comparável**.

Métricas espaciais opcionais:

- `pixel_auroc`
- `pixel_auprc`
- `mask_iou`
- `dice`

Regras:

- essas métricas ficam em uma trilha separada;
- não entram no ranking principal v1;
- métodos sem mapa espacial compatível não devem ser penalizados no leaderboard principal.

## Explicações visuais complementares

Para modelos sem mapa espacial nativo comparável, o repositório deve usar uma trilha separada de **explicações visuais**, nunca confundida com `anomaly_map`.

Na versão atual:

- `PatchCore` e `PaDiM` entram na trilha de `anomaly_maps/`;
- `CutPaste` já entra, na prática, em uma trilha separada de `visual_explanations/`.

Protocolo complementar:

- `docs/protocol_anomaly_visual_explanations.md`

Regra metodológica:

- explicações visuais de `CutPaste` podem apoiar inspeção qualitativa e auditoria humana;
- elas não entram em ranking pixel-level;
- elas não devem ser comparadas diretamente com `pixel_auroc`, `pixel_auprc`, `mask_iou` ou `dice` de métodos com mapa nativo.

## Outputs obrigatórios

## Artefatos por modelo e seed

Raiz canônica:

`runs/anomaly/<dataset>/<version>/<benchmark>/<model>/seed_<seed>/`

Arquivos obrigatórios por seed:

- `model.json`
- `train_history.csv`
- `threshold_selection.json`
- `val_scores.csv`
- `test_scores.csv`
- `val_metrics.json`
- `test_metrics.json`
- `generator_breakdown.csv`
- `anomaly_type_breakdown.csv`
- `severity_breakdown.csv`

Arquivos opcionais:

- `val_examples.md`
- `test_examples.md`
- `anomaly_maps/`

Materialização opcional pós-benchmark:

- `python scripts/render_anomaly_heatmaps.py --models patchcore padim --seeds 42 52 62 --split test`

Saídas esperadas quando o modelo suporta mapa espacial:

- `anomaly_maps/<split>/images/*.png`
- `anomaly_maps/<split>/heatmap_index.csv`
- `anomaly_maps/<split>/summary.json`
- `anomaly_maps/<split>/contact_sheet_top_scores.png`
- `anomaly_maps/<split>/contact_sheet_anomalies.png`

## Schema mínimo de scores por ROI

Os arquivos `val_scores.csv` e `test_scores.csv` devem conter, no mínimo:

- `roi_id`
- `image_id`
- `crop_path`
- `split`
- `label`
- `score`
- `prediction`
- `threshold`
- `source_kind`
- `generator_family`
- `anomaly_type`
- `severity`

Observação:

- para ROIs normais do split oficial, `generator_family`, `anomaly_type` e `severity` podem ficar vazios;
- para ROIs sintéticas, esses campos devem ser herdados de `records.csv`.

## Consolidação oficial

Raiz canônica:

`reports/tables/<dataset>/<version>/anomaly_benchmark/<benchmark>/`

Arquivos obrigatórios:

- `benchmark_results.csv`
- `benchmark_results.json`
- `benchmark_report.md`

## Conteúdo mínimo do relatório final

O `benchmark_report.md` deve incluir:

- descrição curta do protocolo;
- composição de treino, validação e teste;
- tabela principal com `ROI AUROC` e `ROI AUPRC` por modelo;
- tabela thresholded com `F1`, `Precision`, `Recall` e matriz `TP/TN/FP/FN`;
- breakdown por `generator_family`;
- breakdown por `anomaly_type`;
- breakdown por `severity`;
- threshold selecionado em validação para cada modelo;
- limitações conhecidas do benchmark v1.

## Interpretação correta do v1

- o benchmark v1 mede **detecção de anomalia em ROI**, não inspeção completa de cena;
- o leaderboard oficial atual mede desempenho em `gt_crops`, não em `pred_crops`;
- o pack sintético aceito é pequeno e controlado, então o v1 deve ser lido como benchmark inicial defensável, não como conclusão final sobre generalização em campo;
- comparações entre ChatGPT e Gemini entram como análise de robustez do conjunto sintético, não como objetivo principal do benchmark.

## Limitações conhecidas

- o conjunto anômalo aceito é pequeno (`20` imagens);
- as anomalias são sintéticas, não defeitos reais de campo;
- há dependência entre algumas amostras por compartilharem crops de origem;
- a trilha com `pred_crops` ainda não está oficializada nesta versão;
- métodos sem mapa espacial ficam restritos às métricas ROI-level na trilha principal.
- a generalização para campo real ainda depende de avaliação futura com defeitos reais e com `pred_crops`.

## Estado atual do benchmark oficial

- script fino oficial: `python scripts/run_anomaly_benchmark.py`;
- materialização oficial do dataset do benchmark: `data/interim/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/dataset/`;
- raiz oficial de runs: `runs/anomaly/tower_vision/v2026-04-16/anom_benchmark_fair_v1/`;
- consolidação oficial: `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/`;
- `PatchCore` e `PaDiM` rodam via `anomalib 2.2.x`;
- `CutPaste` roda via implementação local do repositório, com encoder `resnet18`, checkpoint `best/last` e early stopping por `val_roi_auroc`;
- as três seeds oficiais `[42, 52, 62]` já foram concluídas para `PatchCore`, `PaDiM` e `CutPaste`;
- o leaderboard oficial v1 já está fechado na trilha `gt_crops`.
- os heatmaps de teste já foram materializados para `PatchCore` e `PaDiM` nas três seeds;
- `CutPaste` continua sem mapa espacial confiável na implementação atual e, por isso, não entra na trilha de heatmaps.
- a trilha correta para `CutPaste` é `visual_explanations/`, não `anomaly_maps/`.
- as `visual_explanations/test/` do `CutPaste` já foram materializadas nas três seeds com `Grad-CAM`.
- a revisão espacial complementar já foi materializada em:
  - `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_review.md`
  - `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_comparison_summary.csv`
  - `reports/figures/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_review/seed_52_localization_comparison_sheet.png`

## O que foi implementado

- materialização reproduzível dos manifests `train.csv`, `val.csv` e `test.csv` a partir do split oficial e do pack sintético aceito;
- persistência canônica por seed/modelo em `runs/anomaly/.../seed_<seed>/`;
- consolidação automática em `reports/tables/.../anomaly_benchmark/...`;
- adapters reais de `PatchCore` e `PaDiM` usando `anomalib`;
- adapter real de `CutPaste` implementado em `src/towervision/anomaly/backends/cutpaste_backend.py`;
- seleção de threshold operacional exclusivamente em `val`, com piso de `recall`;
- breakdowns automáticos por `generator_family`, `anomaly_type` e `severity`;
- suporte a métricas espaciais (`pixel_auroc` e `pixel_auprc`) quando há mapa de anomalia e máscara.

## Composição efetiva da execução v1

- treino: `628` ROIs normais de `isoladores`;
- validação: `129` ROIs normais + `10` ROIs anômalas sintéticas aceitas;
- teste: `147` ROIs normais + `10` ROIs anômalas sintéticas aceitas.

## Resultados oficiais consolidados

| Modelo | Seeds | Val ROI AUROC | Val ROI AUPRC | Val F1 | Val Recall | Test ROI AUROC | Test ROI AUPRC | Test F1 | Test Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CutPaste` | `3/3` | `0.9982 ± 0.0020` | `0.9845 ± 0.0160` | `0.8486 ± 0.0288` | `0.8333 ± 0.1247` | `0.9880 ± 0.0075` | `0.9178 ± 0.0343` | `0.8083 ± 0.0478` | `0.8667 ± 0.0943` |
| `PatchCore` | `3/3` | `0.9189 ± 0.0203` | `0.6771 ± 0.0048` | `0.4277 ± 0.0516` | `0.9000 ± 0.0000` | `0.9517 ± 0.0086` | `0.6401 ± 0.0087` | `0.5072 ± 0.0405` | `0.6667 ± 0.0471` |
| `PaDiM` | `3/3` | `0.9182 ± 0.0355` | `0.4012 ± 0.1510` | `0.5307 ± 0.1149` | `0.9667 ± 0.0471` | `0.9494 ± 0.0069` | `0.5336 ± 0.0817` | `0.5420 ± 0.0330` | `0.6667 ± 0.1700` |

## Interpretação técnica dos resultados

### Vencedor científico do v1

O vencedor oficial do benchmark atual é `CutPaste`.

Justificativa:

- maior `test_roi_auroc`;
- maior `test_roi_auprc`;
- melhor `test_f1`;
- melhor `test_recall`.

Em termos práticos, ele foi o método que melhor separou ROIs normais de ROIs anômalas no conjunto de teste.

### Leitura correta das métricas

- `ROI AUROC`: mede separação entre normal e anômalo ao longo de todos os thresholds possíveis; é a métrica principal do ranking.
- `ROI AUPRC`: mede a qualidade dessa separação em cenário desbalanceado, onde anomalias são minoria; é a métrica secundária do ranking.
- `F1`: resume o equilíbrio entre `precision` e `recall` depois da escolha de um threshold.
- `Precision`: entre tudo que o modelo marcou como anomalia, quanto realmente era anomalia.
- `Recall`: entre todas as anomalias reais, quantas o modelo encontrou.

Nesta fase, a leitura correta é:

- `AUROC` e `AUPRC` definem a comparação científica;
- `F1`, `precision` e `recall` ajudam a interpretar o comportamento operacional.

### O que os resultados significam

- `CutPaste` dominou a comparação científica em `gt_crops`;
- `PatchCore` ficou em segundo lugar nas métricas threshold-free, mas com desempenho operacional inferior ao `CutPaste`;
- `PaDiM` teve `val_recall` mais alto, o que o torna um comparador relevante se a prioridade operacional for minimizar falsos negativos em validação.

### Leitura operacional importante

O critério operacional do protocolo usa threshold escolhido em `val`, com piso de `recall >= 0.90`.

Os resultados mostram:

- `PatchCore` atinge o piso em média;
- `PaDiM` supera o piso em média;
- `CutPaste`, apesar de ser o melhor cientificamente, fica abaixo desse piso em média na validação.

Isso não muda o ranking oficial do benchmark, mas importa para a próxima etapa de sistema.

### Revisão espacial complementar

A revisão espacial com máscaras aceitas do split `test` mostrou um comportamento importante:

- `CutPaste` venceu claramente em métricas ROI-level;
- mas foi o pior método dos três em alinhamento espacial com a região anômala;
- `PaDiM` foi o melhor método em alinhamento espacial;
- `PatchCore` ficou em posição intermediária.

Médias agregadas da revisão espacial (`10` anomalias de `test` x `3` seeds):

| Modelo | Mass Gain vs Area | Peak in Mask | Dice Top 5% |
| --- | ---: | ---: | ---: |
| `PaDiM` | `12.692` | `0.533` | `0.214` |
| `PatchCore` | `3.253` | `0.433` | `0.137` |
| `CutPaste` | `2.388` | `0.000` | `0.058` |

Leitura correta:

- `CutPaste` continua sendo o melhor classificador de ROI;
- mas suas `visual_explanations` não são boas localizadoras de defeito;
- o padrão observado é compatível com uso de pistas globais ou contextuais do crop, e não apenas da anomalia;
- portanto, a vitória de `CutPaste` no v1 deve ser lida como vitória em **detecção de anomalia por ROI**, não como prova de localização espacial confiável.

## Implicações para o projeto

- para benchmark de ROI em `gt_crops`, `CutPaste` é o método de referência atual;
- para a próxima fase do projeto, ainda vale carregar `PaDiM` como comparador orientado a recall;
- a revisão espacial mostrou que o vencedor ROI-level não é, hoje, o melhor método para interpretação espacial;
- por isso, a próxima pergunta experimental relevante passa a ser: "quanto do desempenho atual vem do corpo do isolador e quanto vem do contexto do crop?".

Em outras palavras:

- o próximo passo metodologicamente mais forte é abrir uma trilha de `gt_masked_crops` ou equivalente, usando máscara/segmentação do isolador para suprimir contexto espúrio;
- `CutPaste` deve entrar nela como vencedor científico do v1 em ROI-level;
- `PaDiM` deve permanecer como comparador espacialmente mais defensável;
- a trilha `pred_crops` continua necessária, mas passa a ser o passo seguinte da avaliação ponta a ponta.

## Concessões de implementação já feitas

- `PatchCore` usa `coreset_sampling_ratio = 0.01` na configuração atual para manter o benchmark operacional no hardware disponível sem alterar o contrato experimental;
- essa concessão pode afetar levemente o teto absoluto de desempenho do `PatchCore`, mas não invalida a comparação v1 já executada.

## Relatórios recomendados

- relatório automático canônico: `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/benchmark_report.md`
- interpretação técnica curada: `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/benchmark_interpretation.md`

## Próximo passo após o v1

- congelar o v1 em `gt_crops` como concluído;
- manter a leitura oficial de `CutPaste` como vencedor ROI-level e de `PaDiM` como referência espacial complementar;
- materializar uma trilha de `gt_masked_crops` ou equivalente, usando máscara/segmentação do isolador;
- comparar `gt_crops` vs `gt_masked_crops` para medir redução de shortcut/contexto;
- só depois abrir a trilha `pred_crops`;
- decidir se a pipeline real deve operar em crop puro, crop mascarado ou componente segmentado.
