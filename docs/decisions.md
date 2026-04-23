# Architectural Decisions

## ADR-001: layout `src/`

Toda a lógica do projeto fica em `src/towervision/` para facilitar testes, reuso e instalação local.

## ADR-002: manifests simples

Os manifests intermediários usam JSON simples e legível, com `pathlib` no código e serialização explícita para evitar formatos opacos.

## ADR-003: scripts finos

Os scripts em `scripts/` servem apenas como entrypoints para a pipeline, incluindo DVC e execução manual.

## ADR-004: crops separados

`gt_crops` e `pred_crops` são mantidos separados desde o início para suportar comparação direta no benchmark de anomaly detection.

## ADR-005: placeholders funcionais

Treino e inferência começam como placeholders determinísticos, permitindo validar estrutura, I/O e rastreamento de experimento antes da integração dos modelos reais.

## ADR-006: separar vencedor de benchmark e decisão operacional

O repositório distingue claramente:

- vencedor científico do benchmark de detecção, definido pelo protocolo fechado;
- detector operacional escolhido para alimentar a próxima etapa da pipeline.

Essa separação evita usar o conjunto de teste para escolher o componente operacional do sistema e permite decisões orientadas pela classe crítica quando necessário.

## ADR-007: permitir evolução para sistema híbrido

O sistema pode evoluir para usar detectores diferentes para regiões ou classes diferentes, por exemplo:

- um detector para `torre`;
- outro detector para `isoladores`.

Essa opção é aceita como arquitetura futura, especialmente se o escopo de anomalias incluir defeitos estruturais além de `isoladores`. Quando adotada, deve ser tratada como sistema híbrido explícito, com protocolo e avaliação próprios.

## ADR-008: benchmark de anomalia v1 usa ranking ROI-level e trilha GT oficial

Para a versão atual do benchmark de anomalia:

- o ranking principal deve usar métricas **threshold-free** em nível de ROI;
- a métrica principal oficial é `test_roi_auroc`;
- a métrica secundária oficial é `test_roi_auprc`;
- thresholds operacionais devem ser escolhidos somente em `val`;
- a trilha oficial v1 usa `gt_crops` normais do split congelado e anomalias sintéticas aceitas do pack `anomaly_controlled_v1`;
- a trilha com `pred_crops` é considerada extensão futura e não entra no leaderboard oficial v1.

Essa decisão separa:

- benchmark científico principal, comparável entre métodos heterogêneos;
- análise operacional com threshold calibrado em validação;
- futuras extensões detector-condicionadas, que devem ganhar protocolo próprio quando forem oficializadas.

## ADR-009: adapters reais do benchmark de anomalia

Para a implementação atual do benchmark de anomalia v1:

- `PatchCore` e `PaDiM` usam `anomalib` como backend de referência;
- `CutPaste` usa implementação local e auditável do repositório;
- todos os métodos produzem o mesmo contrato de artefatos em `runs/anomaly/...`;
- smoke runs reais em `seed_42` validam a integração antes da execução completa das seeds oficiais.

## ADR-010: separar anomaly maps nativos de visual explanations

O repositório deve distinguir claramente:

- `anomaly_maps/`
  - mapas espaciais nativos de modelos que os produzem, como `PatchCore` e `PaDiM`;
- `visual_explanations/`
  - explicações visuais complementares de modelos sem mapa espacial nativo comparável, como `CutPaste`.

Essa separação existe para evitar dois erros metodológicos:

- tratar explicação visual como se fosse máscara de defeito;
- comparar mapas nativos e visualizações derivadas como se fossem equivalentes.

Implicação:

- `visual_explanations/` pode apoiar auditoria humana e análise qualitativa;
- `visual_explanations/` não entra no ranking pixel-level nem substitui métricas espaciais oficiais.

## ADR-011: Grad-CAM como explicação visual inicial do CutPaste

Para a trilha complementar de explicabilidade do `CutPaste`, o método inicial adotado é:

- `Grad-CAM`

Justificativa:

- permite inspecionar visualmente quais regiões contribuíram para o score;
- é uma explicação de decisão consolidada e amplamente compreendida;
- evita chamar de "mapa de anomalia" algo que não é equivalente ao mapa nativo de `PatchCore` e `PaDiM`.

Consequência:

- os artefatos de `CutPaste` devem ficar em `visual_explanations/`;
- esses artefatos servem para auditoria qualitativa;
- esses artefatos não entram em ranking espacial oficial.

## ADR-012: separar vencedor ROI-level de confiabilidade espacial

No benchmark atual de anomalia:

- `CutPaste` é o vencedor científico em métricas ROI-level;
- isso não implica que ele seja o melhor método para localização espacial da anomalia;
- revisões qualitativas e quantitativas com máscara mostraram que `CutPaste` é o pior dos três em alinhamento espacial;
- `PaDiM` é hoje o método mais defensável para interpretação espacial complementar.

Consequência:

- o repositório deve manter duas leituras simultâneas:
  - vencedor ROI-level do benchmark;
  - confiança espacial/interpretação visual.

## ADR-013: hipótese prioritária de masked crops por segmentação do isolador

A próxima hipótese forte de pesquisa do projeto é:

- segmentar ou mascarar o isolador;
- aplicar os métodos de anomalia principalmente sobre a região do componente, reduzindo fundo, cabos e ferragem como fonte de shortcut.

Diretriz:

- a primeira trilha a ser aberta após o v1 deve ser `gt_masked_crops` ou equivalente;
- a trilha `pred_crops` continua importante, mas vem depois da checagem de contexto espúrio em `gt_crops`.
