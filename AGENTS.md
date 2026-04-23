# AGENTS

## Objetivo

Este arquivo define como agentes e assistentes devem agir neste repositório.
Ele deve guardar regras estáveis de operação, não estado temporário do projeto.

## Ordem de precedência

1. Pedido explícito do usuário no turno atual.
2. Regras deste `AGENTS.md`.
3. Configurações ativas em `params.yaml` e `configs/`.
4. Protocolos e contratos oficiais em `docs/`.
5. Materiais consultivos em `docs/references/deepsearch/`.

## Rotina obrigatória antes de agir

- Ler este `AGENTS.md`.
- Identificar dataset e versão ativos em `configs/data/base.yaml` e `params.yaml` quando a tarefa envolver dados, splits, treino, inferência ou avaliação.
- Consultar `docs/` antes de alterar protocolos, contratos, benchmark ou convenções de output.
- Consultar `docs/references/deepsearch/` quando houver material relevante ao tema da tarefa.
- Declarar ambiguidades e premissas quando algo não puder ser inferido com segurança.

## Invariantes do repositório

- Nunca colocar lógica importante em `scripts/`.
- Sempre concentrar a lógica principal em `src/towervision/`.
- Scripts devem apenas ler configuração, chamar funções do pacote e persistir resultados.
- Preferir funções pequenas, nomes claros e tipagem simples com `pathlib` e `typing`.
- Evitar dependências novas sem necessidade explícita do fluxo experimental.
- Outputs grandes devem ir para `runs/`, `reports/` ou subpastas apropriadas de `data/`, nunca para `src/` ou `tests/`.
- Manifests e relatórios devem ser serializados em formatos simples e auditáveis, preferencialmente JSON, YAML, CSV ou Markdown.

## Dados e versionamento

- Tratar `data/raw/<dataset_name>/<dataset_version>/` como snapshot imutável.
- Não mover, converter, sobrescrever ou corrigir dataset bruto sem pedido explícito.
- Não alterar arquivos originais do dataset.
- Sempre atualizar `docs/dataset.md` se o formato de dados mudar.
- Sempre documentar incertezas quando a estrutura, o agrupamento ou a semântica do dataset não forem explícitos.

## Splits, benchmark e avaliação

- Todo split oficial deve ser reproduzível e salvo com artefatos auditáveis.
- Mudanças em splits, crops ou métricas exigem testes ou ajuste dos testes existentes.
- Sempre preservar a separação entre `gt_crops` e `pred_crops`.
- Novos benchmarks devem nascer de arquivos em `configs/experiment/` e chaves em `params.yaml`.
- Protocolos oficiais do benchmark devem viver em `docs/`.

## Fontes oficiais e consultivas

- `docs/` contém contratos e protocolos oficiais do repositório.
- `docs/references/deepsearch/` contém contexto complementar, não normativo.
- Em caso de conflito, prevalecem `params.yaml`, `configs/` e os documentos oficiais em `docs/`.

## Não fazer por padrão

- Não treinar modelos sem pedido explícito.
- Não refatorar a arquitetura sem pedido explícito.
- Não reorganizar pastas sem necessidade justificada.
- Não assumir que uma heurística de split elimina totalmente vazamento sem metadados fortes como `tower_id`, `flight_id` ou `campaign_id`.
- Não promover contexto consultivo a protocolo oficial sem documentação explícita.

## Checklist antes de encerrar mudanças

- Confirmar que a lógica nova ficou em `src/`.
- Confirmar que `README.md` e `docs/` continuam coerentes com o fluxo atual quando a mudança afetar uso, dados ou protocolo.
- Executar testes relevantes ou justificar por que não foi possível.
- Documentar riscos, ambiguidades e limitações restantes quando houver.
