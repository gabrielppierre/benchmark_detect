# anomaly_controlled_v1

Pacote controlado de anomalias sintéticas para benchmark não supervisionado.

## Estrutura

- manifest: `manifest.yaml`
- registros: `records.csv`
- imagens geradas:
- `chatgpt`: `generated/chatgpt/`
- `gemini`: `generated/gemini/`
- prompts:
- `chatgpt`: `prompts/chatgpt/`
- `gemini`: `prompts/gemini/`
- máscaras opcionais: `masks/`

- source crops:
- `val`: `source_crops/val/`
- `test`: `source_crops/test/`
- candidatos exportados: `source_candidates.csv`
- shortlist recomendada: `source_shortlist.csv`

## Regras

- Do not move or overwrite the raw dataset.
- Do not mix synthetic files into data/raw.
- Reference source images in records.csv instead of copying the originals.
- Store every generated image with prompt and optional mask for traceability.

## Uso esperado

- cada linha em `records.csv` representa uma imagem sintética aceita ou em revisão;
- `source_candidates.csv` guarda todos os crops exportados de `val` e `test`;
- `source_shortlist.csv` guarda uma shortlist inicial para geração controlada;
- `source_image_path` deve apontar para a imagem real original no dataset versionado;
- `source_crop_path` deve apontar para um crop em `source_crops/<split>/`;
- `pair_id` deve ser igual entre ChatGPT e Gemini quando os dois usarem o mesmo crop base;
- `severity` deve registrar o nível da anomalia simulada;
- `output_image_path` deve apontar para um arquivo dentro de `generated/<generator>/`;
- `prompt_path` deve apontar para um `.md` dentro de `prompts/<generator>/`;
- `mask_path` é opcional e deve ficar em `masks/` quando existir.
