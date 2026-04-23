# Experiment Plan

## Fase 1

- validar o contrato de dados;
- fechar o baseline de splits;
- rodar o scaffold completo com outputs vazios ou mínimos.

## Fase 2

- integrar o primeiro detector real;
- estabilizar métricas e protocolo de inferência;
- revisar custo de geração de `pred_crops`.

## Fase 3

- integrar o primeiro modelo de anomaly detection;
- comparar `gt_crops` versus `pred_crops`;
- consolidar a tabela de benchmark.

## Fase 4

- congelar a interpretação final do benchmark de detecção v1;
- gerar `pred_crops` com o detector operacional escolhido para a etapa seguinte;
- criar um conjunto pequeno de anomalias sintéticas controladas a partir de imagens reais;
- materializar `source_crops`, `source_shortlist.csv` e `source_shortlist_bundle/` para handoff às IAs;
- usar prompt único e rastreabilidade em `records.csv` para comparação controlada entre ChatGPT e Gemini;
- importar as máscaras anotadas para dentro do pack sintético, corrigindo o split de acordo com `records.csv`;
- gerar overlays e uma prancha única para revisão rápida das máscaras;
- congelar `accepted_for_benchmark=true` para as amostras aprovadas;
- rodar o benchmark não supervisionado inicial com PatchCore, PaDiM e CutPaste;
- decidir se a próxima versão da pipeline continua detector único ou evolui para sistema híbrido por região/classe.

## Estado atual

- fases 1 a 4 concluídas até a preparação do pack sintético controlado;
- `anomaly_controlled_v1` já contém imagens geradas, prompts, máscaras importadas, overlays e registros aceitos;
- o protocolo oficial do benchmark não supervisionado v1 já foi definido em `docs/protocol_anomaly_benchmark_fair_v1.md`;
- a configuração central do benchmark v1 já foi definida em `configs/experiment/anom_benchmark_fair_v1.yaml`;
- o runner oficial do benchmark v1 já materializa dataset, jobs e relatórios em `src/towervision/anomaly/` e `scripts/run_anomaly_benchmark.py`;
- `PatchCore` e `PaDiM` já estão integrados via `anomalib`, e `CutPaste` via implementação local do repositório;
- o benchmark oficial `anom_benchmark_fair_v1` já foi concluído nas três seeds `[42, 52, 62]`;
- `CutPaste` venceu o benchmark em `gt_crops`, com `PaDiM` mantendo relevância operacional por `val_recall`;
- a revisão espacial complementar mostrou que `CutPaste` localiza mal a anomalia e provavelmente usa parte do contexto do crop;
- o próximo passo experimental ativo é abrir uma trilha de `gt_masked_crops` ou equivalente, para medir quanto o mascaramento/segmentação do isolador reduz shortcut e alucinação contextual;
- a trilha `pred_crops` passa a ser o passo seguinte da avaliação ponta a ponta.
