# Anomaly Benchmark Fair v1: Interpretação dos Resultados

## Escopo

Este documento interpreta o benchmark oficial `anom_benchmark_fair_v1` para anomaly detection não supervisionada em ROIs de `isoladores`.

Ele complementa, sem substituir:

- o protocolo oficial em `docs/protocol_anomaly_benchmark_fair_v1.md`;
- o relatório consolidado automático em `benchmark_report.md`.

## Como o experimento foi feito

### Dados usados

- dataset ativo: `tower_vision/v2026-04-16`;
- trilha oficial: `gt_crops`;
- pack sintético aceito: `anomaly_controlled_v1`;
- ROI-alvo: `isoladores`.

### Composição do benchmark

- treino: `628` ROIs normais de `isoladores` do split `train`;
- validação: `129` ROIs normais + `10` ROIs sintéticas anômalas;
- teste: `147` ROIs normais + `10` ROIs sintéticas anômalas.

### Modelos comparados

- `PatchCore` via `anomalib`;
- `PaDiM` via `anomalib`;
- `CutPaste` via implementação local do repositório.

### Harmonização

- `input_size = 256`;
- normalização `imagenet`;
- extrator base `resnet18`;
- seeds oficiais: `[42, 52, 62]`.

### Regras de avaliação

- treino apenas com ROIs normais;
- nenhuma anomalia sintética entra no treino;
- ranking científico definido apenas no `test`;
- threshold operacional calibrado apenas no `val`;
- ranking principal: `test_roi_auroc`;
- ranking secundário: `test_roi_auprc`.

## O que cada métrica quer dizer

### ROI AUROC

Mede o quanto o modelo separa bem ROIs normais de ROIs anômalas ao longo de todos os thresholds possíveis.

Leitura prática:

- perto de `1.0`: separação muito boa;
- perto de `0.5`: separação fraca, próxima do acaso.

É a melhor métrica principal para esta fase porque é:

- threshold-free;
- comparável entre métodos;
- adequada para ranking científico.

### ROI AUPRC

Mede a qualidade do ranking das anomalias quando a classe positiva é rara.

Leitura prática:

- quanto maior, melhor;
- é especialmente importante porque o conjunto é dominado por ROIs normais.

Nesta fase, ela é a métrica secundária porque captura melhor o custo de falsos positivos em cenário desbalanceado.

### F1

Equilíbrio entre `precision` e `recall` depois que um threshold já foi escolhido.

Leitura prática:

- alto `F1`: bom compromisso entre achar anomalias e não alarmar demais;
- baixo `F1`: o threshold escolhido ainda produz muitos erros de um dos dois lados.

### Precision

Entre tudo que o modelo marcou como anomalia, quanto realmente era anomalia.

Leitura prática:

- precision alta: poucos falsos positivos;
- precision baixa: muitos alarmes indevidos.

### Recall

Entre todas as anomalias reais, quantas o modelo encontrou.

Leitura prática:

- recall alto: poucos falsos negativos;
- recall baixo: o modelo deixa passar anomalias.

No projeto, recall tem importância operacional alta, porque uma anomalia não detectada não segue para inspeção posterior.

## Resultados oficiais consolidados

### Tabela principal

| Modelo | Seeds | Val ROI AUROC | Val ROI AUPRC | Val F1 | Val Recall | Test ROI AUROC | Test ROI AUPRC | Test F1 | Test Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CutPaste` | `3/3` | `0.9982 ± 0.0020` | `0.9845 ± 0.0160` | `0.8486 ± 0.0288` | `0.8333 ± 0.1247` | `0.9880 ± 0.0075` | `0.9178 ± 0.0343` | `0.8083 ± 0.0478` | `0.8667 ± 0.0943` |
| `PatchCore` | `3/3` | `0.9189 ± 0.0203` | `0.6771 ± 0.0048` | `0.4277 ± 0.0516` | `0.9000 ± 0.0000` | `0.9517 ± 0.0086` | `0.6401 ± 0.0087` | `0.5072 ± 0.0405` | `0.6667 ± 0.0471` |
| `PaDiM` | `3/3` | `0.9182 ± 0.0355` | `0.4012 ± 0.1510` | `0.5307 ± 0.1149` | `0.9667 ± 0.0471` | `0.9494 ± 0.0069` | `0.5336 ± 0.0817` | `0.5420 ± 0.0330` | `0.6667 ± 0.1700` |

## Interpretação científica

### Vencedor do benchmark v1

O vencedor científico do benchmark atual é o `CutPaste`.

Motivos:

- maior `test_roi_auroc`;
- maior `test_roi_auprc`;
- melhor `test_f1`;
- melhor `test_recall` entre os três métodos.

Em termos simples: no conjunto de teste, foi o método que melhor separou ROIs normais de ROIs anômalas e o que melhor sustentou essa separação quando um threshold concreto foi aplicado.

### Leitura por modelo

#### CutPaste

Pontos fortes:

- desempenho claramente superior no teste;
- separação quase perfeita entre normal e anômalo;
- bom `F1` e bom `recall`;
- robustez alta tanto para imagens geradas com ChatGPT quanto com Gemini.

Ponto de atenção:

- o `val_recall` médio ficou abaixo do piso operacional de `0.90`;
- isso indica que, em parte das seeds, o critério com piso de recall precisou recorrer ao fallback de maior `val_f1`.

Leitura correta:

- é o melhor método para ranking científico em `gt_crops`;
- ainda precisa ser observado com cuidado em uso operacional, se o projeto priorizar recall máximo antes de tudo.

#### PatchCore

Pontos fortes:

- desempenho razoável em `ROI AUROC`;
- `ROI AUPRC` superior ao `PaDiM`;
- `val_recall` exatamente no piso operacional.

Pontos fracos:

- `F1` baixo;
- `test_recall` apenas mediano;
- comportamento fraco em `crack` e muito fraco em `localized surface damage`.

Leitura correta:

- separa relativamente bem os casos, mas transforma essa separação em decisões operacionais piores que o `CutPaste`;
- é um baseline sólido, mas não o melhor candidato atual.

#### PaDiM

Pontos fortes:

- maior `val_recall` médio;
- comportamento mais conservador em validação, com menor risco de perder anomalia nessa etapa.

Pontos fracos:

- pior `ROI AUPRC` global;
- desempenho final inferior ao `CutPaste`;
- também ficou fraco em `crack` e em parte de `localized surface damage`.

Leitura correta:

- é o modelo mais orientado a recall em validação;
- não foi o melhor modelo para ranking final nem para equilíbrio geral.

## Interpretação por tipo de anomalia

### CutPaste

Foi muito forte em:

- `localized surface damage`
- `partial chipping`
- `severe contamination`

Teve desempenho mais modesto em:

- `crack`
- `burn mark`

Isso sugere que o modelo responde especialmente bem a alterações mais amplas de textura ou superfície, e menos a defeitos finos e lineares.

### PatchCore

Foi forte em:

- `burn mark`
- `partial chipping`
- `severe contamination`

Foi fraco em:

- `crack`
- `localized surface damage`

Em particular, zerar `recall` em `localized surface damage` é um sinal de fragilidade para defeitos localizados e sutis.

### PaDiM

Manteve desempenho intermediário, mas sem dominar nenhuma categoria.

Seu padrão sugere:

- bom comportamento para recall em validação;
- menor capacidade de ordenar corretamente os casos mais difíceis no teste.

## O que isso implica para o projeto

### Implicação 1: já existe um vencedor claro em `gt_crops`

Para o benchmark científico atual, `CutPaste` deve ser tratado como o melhor método de anomaly detection em ROI de `isoladores`.

### Implicação 2: o próximo teste relevante continua sendo sobre a qualidade da ROI

O benchmark em `gt_crops` já respondeu a pergunta metodológica inicial:

- entre `PatchCore`, `PaDiM` e `CutPaste`, qual método funciona melhor quando a ROI está correta?

A próxima pergunta útil para o projeto agora se divide em duas camadas:

- primeiro: quanto do desempenho atual vem do **isolador** e quanto vem do **contexto do crop**;
- depois: o que acontece quando a ROI vem do detector de componentes, e não do ground truth.

Por isso, a sequência correta passa a ser:

- `gt_crops -> anomaly model`
- `gt_masked_crops -> anomaly model`
- `pred_crops -> anomaly model`

### Implicação 3: a revisão espacial muda a leitura do vencedor

A revisão espacial complementar mostrou que:

- `CutPaste` é o melhor método em **classificação de ROI**;
- mas não é um bom método para **localização espacial da anomalia**;
- o padrão observado é compatível com dependência de contexto ou shortcut visual, e não apenas da falha em si.

Artefatos canônicos dessa revisão:

- `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_review.md`
- `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_comparison_summary.csv`
- `reports/figures/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/localization_review/seed_52_localization_comparison_sheet.png`

Resumo quantitativo:

- `PaDiM`: `mass_gain_vs_area = 12.692`, `peak_in_mask = 0.533`, `dice_top5 = 0.214`
- `PatchCore`: `3.253`, `0.433`, `0.137`
- `CutPaste`: `2.388`, `0.000`, `0.058`

Leitura correta:

- o `CutPaste` não venceu "por sorte";
- mas ele pode ter vencido usando o **sinal errado**, isto é, correlações globais ou contextuais do crop;
- por isso, ele deve continuar como vencedor ROI-level do benchmark, mas ainda não como escolha final e incontestável do sistema.

### Implicação 4: o melhor método científico não elimina análise operacional

Mesmo com vitória clara do `CutPaste`, o projeto ainda precisa distinguir:

- melhor método de benchmark;
- melhor método para pipeline real.

Se o sistema real priorizar não perder anomalias acima de qualquer outra coisa, o `PaDiM` ainda merece permanecer como comparador na próxima fase por causa do seu `val_recall`.

### Implicação 5: o próximo experimento forte é reduzir contexto

Dado o padrão espacial observado, o próximo experimento mais valioso deixa de ser apenas `pred_crops`.

A hipótese prioritária agora é:

- segmentar ou mascarar o isolador;
- gerar uma trilha de `gt_masked_crops` ou equivalente;
- medir se isso reduz shortcut/contexto e melhora a coerência espacial dos modelos.

Só depois disso a trilha `pred_crops` deve ser aberta como avaliação ponta a ponta do sistema real.

### Implicação 6: a taxonomia atual ainda é pequena

Este v1 é defensável, mas ainda restrito:

- apenas `20` amostras sintéticas aceitas;
- apenas `isoladores`;
- apenas trilha `gt_crops`;
- anomalias sintéticas e não defeitos reais de campo.

Portanto, o resultado é forte como benchmark interno inicial, mas ainda não deve ser tratado como prova final de generalização em produção.

## Limitações que continuam valendo

- conjunto anômalo pequeno;
- anomalias sintéticas, não reais;
- dependência parcial entre amostras derivadas dos mesmos crops fonte;
- severidade ainda concentrada em uma faixa visual relativamente estreita;
- ausência, nesta fase, da avaliação ponta a ponta com `pred_crops`.

## Conclusão

O benchmark `anom_benchmark_fair_v1` já permite uma conclusão clara:

- `CutPaste` é o melhor método atual para anomaly detection em `gt_crops` de `isoladores`;
- `PatchCore` e `PaDiM` ficam atrás com margem relevante;
- `PaDiM` ainda preserva interesse operacional como comparador orientado a recall;
- a revisão espacial mostrou que o `CutPaste` não localiza bem a falha e pode estar usando contexto;
- o próximo experimento correto é verificar essa hipótese com `gt_masked_crops` antes da abertura da trilha `pred_crops`.
