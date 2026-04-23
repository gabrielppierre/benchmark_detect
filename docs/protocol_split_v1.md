# Split Protocol V1

## Objetivo

Fechar um split oficial reproduzível para `tower_vision/v2026-04-16`, preservando as duas classes supervisionadas `torre` e `isoladores` e reduzindo vazamento entre frames muito próximos.

## Heurística de group_id

- origem do group_id: nome do arquivo no padrão `DJI_YYYYMMDDHHMMSS_SEQ_V`;
- metadado usado: timestamp de captura embutido no nome;
- regra: agrupar imagens em buckets temporais contíguos de 30 segundos;
- formato do group_id: `YYYYMMDD_HHMM_SS_30s`.

## Estratégia de split

- ordenar todos os grupos cronologicamente;
- escolher duas fronteiras contíguas `train | val | test`;
- otimizar as fronteiras para aproximar `70/15/15` em número de imagens;
- não embaralhar grupos entre partições.

## Motivo da escolha

- `minute-level` era conservador demais e deixava poucos grupos;
- grupos por imagem ou por segundo eram finos demais e aumentavam risco de vazamento;
- buckets de 30 segundos mantêm blocos temporais razoavelmente coesos e ainda permitem um split utilizável.

## Artefatos esperados

- `data/splits/tower_vision/v2026-04-16/splits.json`
- `data/splits/tower_vision/v2026-04-16/split_metadata.json`
- `data/splits/tower_vision/v2026-04-16/split_distribution.json`
- `data/splits/tower_vision/v2026-04-16/split_distribution.md`
- `data/splits/tower_vision/v2026-04-16/samples/`

## Incertezas

- o dataset atual parece vir de uma única sequência contínua de captura;
- não há `tower_id`, `flight_id` ou `campaign_id` explícitos para separação forte;
- o split reduz vazamento temporal, mas não elimina totalmente a chance de cenas muito parecidas em buckets vizinhos.
