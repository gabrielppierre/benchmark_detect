# YOLO11s Verdict

## Escopo

- Modelo: `YOLO11s`
- Dataset: `tower_vision/v2026-04-16`
- Split: `official_v1`
- Seeds avaliadas: `42`, `52`, `62`
- Classe crítica: `isoladores`
- Critério de checkpoint: maior `val_mAP50_95`

## Resultado Consolidado

| Métrica | Média | Desvio | Mínimo | Máximo |
| --- | ---: | ---: | ---: | ---: |
| `val_mAP50_95` | 0.7910 | 0.0022 | 0.7884 | 0.7938 |
| `val_AP50_95_isoladores` | 0.7229 | 0.0015 | 0.7213 | 0.7249 |
| `val_Recall_isoladores` | 0.9974 | 0.0037 | 0.9922 | 1.0000 |
| `test_mAP50_95` | 0.8058 | 0.0090 | 0.7952 | 0.8172 |
| `test_AP50_95_isoladores` | 0.7098 | 0.0088 | 0.6985 | 0.7199 |
| `test_Recall_isoladores` | 0.9752 | 0.0059 | 0.9670 | 0.9803 |
| `test_AP50_95_torre` | 0.9017 | 0.0094 | 0.8920 | 0.9144 |
| `test_Recall_torre` | 0.9899 | 0.0000 | 0.9899 | 0.9899 |

## Convergência

| Seed | Melhor época | Épocas rodadas | Parada após melhor |
| ---: | ---: | ---: | ---: |
| 42 | 66 | 86 | 20 |
| 52 | 43 | 63 | 20 |
| 62 | 33 | 53 | 20 |

Observações:

- As três execuções respeitaram `patience = 20`.
- As perdas de treino caíram de forma consistente até o fim.
- Não há sinal evidente de instabilidade, NaN, colapso ou overfitting severo nas curvas.
- A variação entre seeds é baixa para as métricas principais, especialmente em validação.

## Diagnóstico Por Threshold

Thresholds abaixo são diagnóstico operacional. O limiar final para recorte deve ser escolhido em validação, sem usar teste para tuning.

### Validação, agregado em 3 seeds

| Confiança | Classe | TP | FP | FN | Duplicatas FP | Precisão | Recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | torre | 255 | 3 | 0 | 0 | 0.9884 | 1.0000 |
| 0.25 | isoladores | 387 | 15 | 0 | 14 | 0.9627 | 1.0000 |
| 0.30 | torre | 255 | 3 | 0 | 0 | 0.9884 | 1.0000 |
| 0.30 | isoladores | 387 | 9 | 0 | 9 | 0.9773 | 1.0000 |
| 0.40 | torre | 255 | 3 | 0 | 0 | 0.9884 | 1.0000 |
| 0.40 | isoladores | 384 | 4 | 3 | 4 | 0.9897 | 0.9922 |
| 0.50 | torre | 255 | 3 | 0 | 0 | 0.9884 | 1.0000 |
| 0.50 | isoladores | 380 | 1 | 7 | 1 | 0.9974 | 0.9819 |

### Teste, agregado em 3 seeds

| Confiança | Classe | TP | FP | FN | Duplicatas FP | Precisão | Recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | torre | 294 | 0 | 3 | 0 | 1.0000 | 0.9899 |
| 0.25 | isoladores | 441 | 55 | 0 | 44 | 0.8891 | 1.0000 |
| 0.30 | torre | 294 | 0 | 3 | 0 | 1.0000 | 0.9899 |
| 0.30 | isoladores | 441 | 46 | 0 | 38 | 0.9055 | 1.0000 |
| 0.40 | torre | 294 | 0 | 3 | 0 | 1.0000 | 0.9899 |
| 0.40 | isoladores | 441 | 34 | 0 | 30 | 0.9284 | 1.0000 |
| 0.50 | torre | 294 | 0 | 3 | 0 | 1.0000 | 0.9899 |
| 0.50 | isoladores | 438 | 19 | 3 | 19 | 0.9584 | 0.9932 |

Interpretação:

- Para recorte de ROIs, `conf >= 0.30` é um candidato inicial defensável por manter `Recall_isoladores = 1.0` em validação e reduzir falsos positivos em relação a 0.25.
- `conf >= 0.40` reduz ainda mais falsos positivos, mas já introduz alguns falsos negativos em validação.
- Como a etapa de anomalia depende de não perder isoladores, o ponto inicial mais conservador é `0.30`.

## Revisão Visual

Artefatos revisados:

- `results.png`
- `BoxPR_curve.png`
- `BoxR_curve.png`
- `confusion_matrix_normalized.png`
- `test_eval/val_batch*_pred.jpg`
- `test_eval/val_batch*_labels.jpg`

Achados:

- As torres são detectadas de forma consistente, com caixas estáveis entre seeds.
- Os isoladores principais próximos à torre são detectados de forma consistente.
- Não foi observado padrão forte de falso negativo visual nos batches revisados.
- Há falsos positivos ou duplicatas de `isoladores` com confiança baixa a média, principalmente entre `0.3` e `0.5`.
- As duplicatas aparecem como o principal problema operacional para geração de crops.
- Algumas caixas de `isoladores` são estreitas e altas, coerentes com a anotação atual, mas devem ser verificadas antes de usar diretamente para anomaly detection.

## Riscos

- O dataset é pequeno e o split ainda depende de heurística temporal, não de `tower_id` ou `flight_id` forte.
- As imagens do teste são visualmente muito parecidas entre si, então o bom resultado pode não representar variação ampla de campo.
- A revisão visual disponível é por mosaicos do Ultralytics; para auditoria final, é melhor gerar uma galeria dedicada de TP, FP, FN e duplicatas.
- O relatório consolidado `benchmark_report.md` está incompleto no momento: mostra `2/2`, porque foi gerado a partir de uma execução filtrada das seeds `52` e `62`. Os três `result.json` existem e foram usados nesta avaliação.

## Veredito

`YOLO11s` é um baseline forte e operacionalmente viável para a versão atual do dataset.

Para a etapa futura de anomaly detection, o modelo passa com folga no critério principal: `val_Recall_isoladores` médio de `0.9974` e `test_Recall_isoladores` médio de `0.9752`.

O modelo ainda não deve ser declarado vencedor final do benchmark porque faltam os demais detectores, mas já é adequado como baseline de referência e como primeiro gerador de ROIs, desde que o limiar de confiança e a política de remoção de duplicatas sejam definidos em validação.
