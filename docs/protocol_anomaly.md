# Anomaly Protocol

## Objetivo

Comparar modelos não supervisionados de anomaly detection sobre ROIs estruturais, começando por `isoladores`, distinguindo treino em ROIs de ground truth e avaliação em ROIs de ground truth e predição.

## Unidade de comparação

- treino padrão em `data/rois/<dataset>/<version>/gt_crops/`;
- avaliação comparativa em `gt_crops` e `pred_crops`;
- score por ROI como saída primária;
- nesta fase, a unidade inicial continua sendo ROI de `isoladores`, mas ROIs de `torre` podem ser incorporadas em versões futuras para defeitos estruturais mais amplos.

## Modelos previstos

- PatchCore
- PaDiM
- CutPaste

## Protocolo oficial atual

- benchmark oficial atual: `anom_benchmark_fair_v1`
- config central: `configs/experiment/anom_benchmark_fair_v1.yaml`
- protocolo específico: `docs/protocol_anomaly_benchmark_fair_v1.md`
- protocolo complementar de explicabilidade: `docs/protocol_anomaly_visual_explanations.md`
- runner oficial atual: `python scripts/run_anomaly_benchmark.py`

## Próximo passo oficial

- congelar o benchmark justo de detecção v1 como concluído;
- consumir o pack sintético aceito `anomaly_controlled_v1` como conjunto controlado já fechado para a próxima fase;
- manter `CutPaste` como vencedor científico atual do v1 em `gt_crops`, mas não como localizador espacial confiável;
- manter `PaDiM` como comparador espacialmente mais defensável na próxima fase;
- abrir primeiro uma trilha de `gt_masked_crops` ou equivalente, usando máscara/segmentação do isolador para reduzir contexto espúrio;
- comparar desempenho dos modelos em `gt_crops` versus `gt_masked_crops`;
- só depois abrir a trilha `pred_crops` para medir degradação ponta a ponta do sistema real.

## Protocolo inicial de anomalias sintéticas controladas

- partir de imagens reais já presentes no dataset versionado;
- gerar um conjunto pequeno e auditável de imagens com anomalias sintéticas;
- plano inicial: `5` imagens geradas com ChatGPT e `5` imagens geradas com Gemini, sempre a partir de imagens reais do projeto;
- localização canônica inicial: `data/synthetic/tower_vision/v2026-04-16/anomaly_controlled_v1/`;
- manter rastreabilidade de origem: imagem base, ferramenta geradora, prompt, região alterada e tipo de anomalia simulada;
- não misturar esse material ao dataset bruto original;
- salvar esse conjunto em subpasta versionada própria, separada do snapshot bruto, para permitir benchmark controlado e reversível.

## Operação do pacote sintético controlado

- pack inicial ativo: `data/synthetic/tower_vision/v2026-04-16/anomaly_controlled_v1/`;
- inicialização reproduzível: `python scripts/init_synthetic_anomaly_pack.py`;
- export de source crops: `python scripts/prepare_synthetic_source_crops.py`;
- materialização da pasta de handoff: `python scripts/materialize_synthetic_shortlist.py`;
- sincronização inicial de `records.csv`: `python scripts/sync_synthetic_records.py`;
- importação das máscaras anotadas no Roboflow: `python scripts/import_roboflow_masks.py --export-root dataset_segmentation_roboflow`;
- overlays para revisão rápida das máscaras: `python scripts/render_synthetic_mask_overlays.py`;
- aceite final dos registros curados: `python scripts/accept_synthetic_records.py`;
- os insumos para as IAs devem vir de `source_crops/`, não da imagem completa;
- padrão inicial de export: `isoladores` de `val` e `test`, com padding leve para preservar contexto local;
- imagens geradas pelo ChatGPT devem ficar em `generated/chatgpt/`;
- imagens geradas pelo Gemini devem ficar em `generated/gemini/`;
- prompts e instruções usados em cada geração devem ficar em `prompts/chatgpt/` e `prompts/gemini/`;
- máscaras opcionais de edição localizada devem ficar em `masks/`;
- `source_candidates.csv` deve guardar todos os crops exportados elegíveis;
- `source_shortlist.csv` deve guardar a shortlist recomendada para a primeira rodada;
- `source_shortlist_bundle/` deve guardar uma cópia única dos crops da shortlist para envio às IAs;
- `records.csv` deve ser preenchido linha a linha com `pair_id`, `source_image_id`, `source_image_path`, `source_crop_path`, `source_split`, gerador, modelo, tipo de anomalia, severidade, caminho do arquivo gerado e caminho do prompt;
- antes da anotação manual, `records.csv` pode ser pré-preenchido automaticamente com os caminhos dos outputs gerados e marcado como `pending_annotation`;
- quando o Roboflow exportar tudo em `train/`, esse split deve ser ignorado na importação; a fonte de verdade para `val` e `test` continua sendo `records.csv` e o `pair_id` já congelado no pack;
- a imagem real de origem não deve ser copiada para dentro do pack; ela deve ser apenas referenciada;
- somente imagens sintéticas aceitas para benchmark devem receber `accepted_for_benchmark=true`.

## Estado atual implementado

- o pack sintético controlado foi inicializado e está materializado em `data/synthetic/tower_vision/v2026-04-16/anomaly_controlled_v1/`;
- foram exportados source crops apenas de `isoladores` e apenas de `val` e `test`;
- foi usado `padding=64` para preservar contexto local sem recorrer à imagem completa;
- foram exportados `129` source crops de `val` e `147` source crops de `test`;
- `source_candidates.csv` lista todos os crops elegíveis para geração sintética;
- `source_shortlist.csv` lista uma shortlist inicial com `10` crops, balanceada temporalmente por partição;
- `source_shortlist_bundle/` contém uma cópia única desses `10` crops para envio direto ao ChatGPT e ao Gemini;
- foram geradas `20` imagens sintéticas no total, sendo `10` do ChatGPT e `10` do Gemini;
- `records.csv` foi sincronizado com a ordem do `source_shortlist_bundle/`, preenchendo `pair_id`, `source_crop_path`, `source_split`, `anomaly_type`, `severity` e `prompt_path`;
- foi feita uma revisão visual das `20` imagens, registrando em `records.csv` sugestões de severidade e comentários de plausibilidade;
- as máscaras anotadas foram exportadas do Roboflow em `COCO Segmentation`, importadas para dentro do pack e corrigidas para os splits oficiais `val` e `test` com base em `records.csv`, sem usar o split `train` do export;
- o campo `mask_path` de `records.csv` agora aponta para arquivos em `masks/val/` e `masks/test/`;
- foram geradas overlays individuais em `reports/figures/tower_vision/v2026-04-16/anomaly_controlled_v1/mask_overlays/`;
- foi gerada uma prancha única de revisão em `reports/figures/tower_vision/v2026-04-16/anomaly_controlled_v1/mask_overlays/contact_sheet.png`;
- as `20` imagens foram marcadas com `accepted_for_benchmark=true` após curadoria visual e validação das máscaras;
- o diretório temporário de export do Roboflow pode ser removido após a importação, pois a fonte de verdade passa a ser o conteúdo já internalizado no pack sintético.
- o runner oficial do benchmark de anomalia já materializa dataset, jobs e relatórios em `src/towervision/anomaly/` e `scripts/run_anomaly_benchmark.py`;
- `PatchCore` e `PaDiM` já estão integrados via `anomalib`, e `CutPaste` via implementação local auditável;
- o benchmark oficial `anom_benchmark_fair_v1` já foi concluído nas três seeds `[42, 52, 62]` para as três famílias;
- o relatório consolidado oficial aponta `CutPaste` como vencedor científico do v1 em `gt_crops`;
- `PaDiM` permaneceu relevante como comparador com `val_recall` mais alto;
- a interpretação técnica consolidada está em `reports/tables/tower_vision/v2026-04-16/anomaly_benchmark/anom_benchmark_fair_v1/benchmark_interpretation.md`.
- `PatchCore` e `PaDiM` já possuem `anomaly_maps/test/` materializados;
- `CutPaste` não possui mapa espacial nativo, mas já possui `visual_explanations/test/` materializadas em trilha separada;
- a trilha complementar de explicabilidade está documentada em `docs/protocol_anomaly_visual_explanations.md`.
- a revisão espacial complementar mostrou que `CutPaste` é forte em ROI-level, mas fraco em localização espacial da anomalia;
- a hipótese de pesquisa agora priorizada é usar máscara/segmentação do isolador para reduzir shortcut/contexto antes da etapa de anomaly detection.

## Estratégia operacional adotada

- a entrada para as IAs deve ser o crop do `isolador`, não a imagem completa;
- o motivo é concentrar resolução útil no objeto de interesse e reduzir variação irrelevante de fundo;
- a geração sintética inicial continua restrita a anomalias de `isoladores`;
- o mesmo modelo de prompt deve ser usado no ChatGPT e no Gemini;
- quando o objetivo for comparar os geradores, o ideal é usar o mesmo `source_crop` nas duas IAs e manter o mesmo `anomaly_type` e `severity`;
- para esta versão do fluxo, assumir que ChatGPT e Gemini serão usados sem campo separado de `negative prompt`;
- por isso, as restrições negativas devem ser incorporadas diretamente ao prompt principal, no bloco `Negative constraints`.

## Negative Constraints Embutidas

Usar este bloco diretamente no prompt principal:

```text
Negative constraints:
Do not change the background, tower, cables, sky, framing, aspect ratio, camera angle, lighting setup, or overall scene composition. Do not add text, watermark, logo, extra objects, multiple anomalies, unrealistic textures, painterly style, CGI appearance, or global image degradation.
```

## Prompts Oficiais da Primeira Rodada

Modelo único para ChatGPT e Gemini, com uma anomalia por imagem e escopo restrito ao `isolador`.

### Prompt 1: crack

```text
You are editing a real inspection photo crop of a power-line insulator.

Task:
Insert exactly one realistic visual anomaly into the insulator only.

Primary goal:
Create a photorealistic edited version of the input image that still looks like a real inspection crop, while adding one localized anomaly of type crack with severity moderate.

Hard constraints:
- Keep the same framing, aspect ratio, resolution, perspective, and camera viewpoint.
- Preserve the same background, lighting, shadows, color balance, and overall photographic style.
- Preserve the global geometry and identity of the insulator.
- Edit only a small localized region of the insulator.
- Do not add any new objects outside the insulator.
- Do not change the tower, cables, sky, background, or scene composition.
- Do not create multiple defects.
- Do not turn the image into an illustration, CGI render, painting, or synthetic-looking artwork.
- Do not degrade the whole image.
- The result must remain visually plausible for a technical inspection scenario.

Anomaly specification:
- anomaly scope: insulator only
- anomaly type: crack
- severity: moderate

Desired behavior:
- The anomaly must be clearly visible but still plausible.
- The anomaly must affect only one limited area of the insulator.
- The rest of the insulator should remain intact.
- Material appearance must remain coherent with the apparent insulator material in the image.
- Keep texture realism and local consistency.

Negative constraints:
Do not change the background, tower, cables, sky, framing, aspect ratio, camera angle, lighting setup, or overall scene composition. Do not add text, watermark, logo, extra objects, multiple anomalies, unrealistic textures, painterly style, CGI appearance, or global image degradation.

Output:
Return one edited image only.
```

### Prompt 2: partial chipping

```text
You are editing a real inspection photo crop of a power-line insulator.

Task:
Insert exactly one realistic visual anomaly into the insulator only.

Primary goal:
Create a photorealistic edited version of the input image that still looks like a real inspection crop, while adding one localized anomaly of type partial chipping with severity moderate.

Hard constraints:
- Keep the same framing, aspect ratio, resolution, perspective, and camera viewpoint.
- Preserve the same background, lighting, shadows, color balance, and overall photographic style.
- Preserve the global geometry and identity of the insulator.
- Edit only a small localized region of the insulator.
- Do not add any new objects outside the insulator.
- Do not change the tower, cables, sky, background, or scene composition.
- Do not create multiple defects.
- Do not turn the image into an illustration, CGI render, painting, or synthetic-looking artwork.
- Do not degrade the whole image.
- The result must remain visually plausible for a technical inspection scenario.

Anomaly specification:
- anomaly scope: insulator only
- anomaly type: partial chipping
- severity: moderate

Desired behavior:
- The anomaly must be clearly visible but still plausible.
- The anomaly must affect only one limited area of the insulator.
- The rest of the insulator should remain intact.
- Material appearance must remain coherent with the apparent insulator material in the image.
- Keep texture realism and local consistency.

Negative constraints:
Do not change the background, tower, cables, sky, framing, aspect ratio, camera angle, lighting setup, or overall scene composition. Do not add text, watermark, logo, extra objects, multiple anomalies, unrealistic textures, painterly style, CGI appearance, or global image degradation.

Output:
Return one edited image only.
```

### Prompt 3: burn mark

```text
You are editing a real inspection photo crop of a power-line insulator.

Task:
Insert exactly one realistic visual anomaly into the insulator only.

Primary goal:
Create a photorealistic edited version of the input image that still looks like a real inspection crop, while adding one localized anomaly of type burn mark with severity moderate.

Hard constraints:
- Keep the same framing, aspect ratio, resolution, perspective, and camera viewpoint.
- Preserve the same background, lighting, shadows, color balance, and overall photographic style.
- Preserve the global geometry and identity of the insulator.
- Edit only a small localized region of the insulator.
- Do not add any new objects outside the insulator.
- Do not change the tower, cables, sky, background, or scene composition.
- Do not create multiple defects.
- Do not turn the image into an illustration, CGI render, painting, or synthetic-looking artwork.
- Do not degrade the whole image.
- The result must remain visually plausible for a technical inspection scenario.

Anomaly specification:
- anomaly scope: insulator only
- anomaly type: burn mark
- severity: moderate

Desired behavior:
- The anomaly must be clearly visible but still plausible.
- The anomaly must affect only one limited area of the insulator.
- The rest of the insulator should remain intact.
- Material appearance must remain coherent with the apparent insulator material in the image.
- Keep texture realism and local consistency.

Negative constraints:
Do not change the background, tower, cables, sky, framing, aspect ratio, camera angle, lighting setup, or overall scene composition. Do not add text, watermark, logo, extra objects, multiple anomalies, unrealistic textures, painterly style, CGI appearance, or global image degradation.

Output:
Return one edited image only.
```

### Prompt 4: severe contamination

```text
You are editing a real inspection photo crop of a power-line insulator.

Task:
Insert exactly one realistic visual anomaly into the insulator only.

Primary goal:
Create a photorealistic edited version of the input image that still looks like a real inspection crop, while adding one localized anomaly of type severe contamination with severity moderate.

Hard constraints:
- Keep the same framing, aspect ratio, resolution, perspective, and camera viewpoint.
- Preserve the same background, lighting, shadows, color balance, and overall photographic style.
- Preserve the global geometry and identity of the insulator.
- Edit only a small localized region of the insulator.
- Do not add any new objects outside the insulator.
- Do not change the tower, cables, sky, background, or scene composition.
- Do not create multiple defects.
- Do not turn the image into an illustration, CGI render, painting, or synthetic-looking artwork.
- Do not degrade the whole image.
- The result must remain visually plausible for a technical inspection scenario.

Anomaly specification:
- anomaly scope: insulator only
- anomaly type: severe contamination
- severity: moderate

Desired behavior:
- The anomaly must be clearly visible but still plausible.
- The anomaly must affect only one limited area of the insulator.
- The rest of the insulator should remain intact.
- Material appearance must remain coherent with the apparent insulator material in the image.
- Keep texture realism and local consistency.

Negative constraints:
Do not change the background, tower, cables, sky, framing, aspect ratio, camera angle, lighting setup, or overall scene composition. Do not add text, watermark, logo, extra objects, multiple anomalies, unrealistic textures, painterly style, CGI appearance, or global image degradation.

Output:
Return one edited image only.
```

### Prompt 5: localized surface damage

```text
You are editing a real inspection photo crop of a power-line insulator.

Task:
Insert exactly one realistic visual anomaly into the insulator only.

Primary goal:
Create a photorealistic edited version of the input image that still looks like a real inspection crop, while adding one localized anomaly of type localized surface damage with severity moderate.

Hard constraints:
- Keep the same framing, aspect ratio, resolution, perspective, and camera viewpoint.
- Preserve the same background, lighting, shadows, color balance, and overall photographic style.
- Preserve the global geometry and identity of the insulator.
- Edit only a small localized region of the insulator.
- Do not add any new objects outside the insulator.
- Do not change the tower, cables, sky, background, or scene composition.
- Do not create multiple defects.
- Do not turn the image into an illustration, CGI render, painting, or synthetic-looking artwork.
- Do not degrade the whole image.
- The result must remain visually plausible for a technical inspection scenario.

Anomaly specification:
- anomaly scope: insulator only
- anomaly type: localized surface damage
- severity: moderate

Desired behavior:
- The anomaly must be clearly visible but still plausible.
- The anomaly must affect only one limited area of the insulator.
- The rest of the insulator should remain intact.
- Material appearance must remain coherent with the apparent insulator material in the image.
- Keep texture realism and local consistency.

Negative constraints:
Do not change the background, tower, cables, sky, framing, aspect ratio, camera angle, lighting setup, or overall scene composition. Do not add text, watermark, logo, extra objects, multiple anomalies, unrealistic textures, painterly style, CGI appearance, or global image degradation.

Output:
Return one edited image only.
```

## Próximo passo técnico

- selecionar as imagens reais base que serão usadas como origem;
- gerar o primeiro lote controlado com `5` amostras via ChatGPT e `5` via Gemini;
- revisar visualmente consistência, realismo e rastreabilidade de cada amostra;
- preencher `records.csv` e congelar o primeiro pack sintético aceito;
- depois disso, materializar o benchmark não supervisionado comparando treino em ROIs normais e avaliação em ROIs normais versus ROIs com anomalias sintéticas controladas.

## Escopo funcional previsto

- escopo inicial: anomalias em `isoladores`;
- escopo futuro aceitável: anomalias estruturais mais amplas, como treliça invertida, ferrugem, empenamento e outros defeitos observáveis na torre;
- por isso, ROIs de `torre` devem ser tratadas como insumo potencial para versões futuras do benchmark de anomalia, mesmo que a primeira rodada continue focada em `isoladores`.

## Saídas esperadas

- artefato do modelo em `runs/anomaly/<experimento>/model.json`;
- resumo de scores em `reports/tables/pipeline_metrics.json`;
- comparação textual em `reports/benchmark_v1.md`.
