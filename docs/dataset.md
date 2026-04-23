# Dataset Contract

## Diretórios

- bruto versionado: `data/raw/<dataset_name>/<dataset_version>/`
- arquivo original: `data/raw/<dataset_name>/<dataset_version>/archive/`
- dataset extraído: `data/raw/<dataset_name>/<dataset_version>/extracted/<source_name>/`
- manifest da versão: `data/raw/<dataset_name>/<dataset_version>/manifest.yaml`
- manifests limpos: `data/interim/<dataset_name>/<dataset_version>/cleaned/images.json` e `annotations.json`
- splits: `data/splits/<dataset_name>/<dataset_version>/splits.json`
- crops: `data/rois/<dataset_name>/<dataset_version>/gt_crops/` e `pred_crops/`
- pacotes sintéticos controlados: `data/synthetic/<dataset_name>/<dataset_version>/<synthetic_pack>/`

## Formato esperado das imagens

As imagens podem estar em subpastas dentro de `data/raw/<dataset_name>/<dataset_version>/extracted/<source_name>/`. As extensões aceitas são definidas em `params.yaml` e `configs/data/base.yaml`.

## Formato esperado das anotações

O projeto agora suporta dois cenários de ingestão:

1. JSON normalizado próprio, com lista de anotações ou chave `annotations`;
2. COCO detection, com `images`, `annotations` e `categories`.

O arquivo `annotations.json` pode ser:

1. uma lista de objetos de anotação;
2. um objeto com a chave `annotations`.

Cada anotação deve seguir o contrato:

```json
{
  "id": "ann-0001",
  "image_id": "image-0001",
  "bbox": [100, 120, 80, 60],
  "label": "isolator"
}
```

Para COCO, o loader preserva o arquivo bruto e normaliza os manifests intermediários para o contrato interno do projeto.

### Regras

- `image_id` deve existir no manifest de imagens;
- `bbox` usa o formato `[x, y, width, height]`;
- largura e altura devem ser positivas;
- coordenadas negativas não são aceitas;
- `label` é opcional e assume `isolator` quando ausente.

## Manifests limpos

`images.json` guarda metadados mínimos por imagem:

```json
{
  "id": "DJI_20250911105044_0101_V",
  "path": "data/raw/tower_vision/v2026-04-16/extracted/imagens_torres_300/images/default/DJI_20250911105044_0101_V.jpg",
  "width": 1920,
  "height": 1080,
  "split": null,
  "metadata": {
    "source_image_id": 1,
    "file_name": "DJI_20250911105044_0101_V.jpg"
  }
}
```

`annotations.json` guarda anotações normalizadas:

```json
{
  "id": "1",
  "image_id": "DJI_20250911105044_0101_V",
  "bbox": [3188.6, 905.4, 650.0, 1586.3],
  "label": "torre",
  "score": null,
  "source": "gt",
  "metadata": {
    "category_id": 1,
    "area": 1031094.9999999998,
    "iscrowd": 0
  }
}
```

Se esse contrato mudar, atualize este documento antes de alterar a pipeline.

## Anomalias sintéticas controladas

Material sintético para benchmark de anomalia deve ficar fora de `data/raw/` e fora do snapshot bruto.

Layout canônico:

```text
data/synthetic/<dataset_name>/<dataset_version>/<synthetic_pack>/
├── manifest.yaml
├── records.csv
├── source_candidates.csv
├── source_shortlist.csv
├── README.md
├── source_crops/
│   ├── val/
│   └── test/
├── generated/
│   ├── chatgpt/
│   └── gemini/
├── prompts/
│   ├── chatgpt/
│   └── gemini/
└── masks/
```

Regras:

- não copiar nem alterar o dataset bruto para dentro desse pacote;
- exportar source crops apenas a partir de `val` e `test`, nunca de `train`;
- usar `source_crops/` como insumo para ChatGPT e Gemini, não a cena completa;
- registrar todos os crops elegíveis em `source_candidates.csv`;
- registrar a shortlist recomendada para geração em `source_shortlist.csv`;
- referenciar a imagem de origem em `records.csv` por `source_image_id` e `source_image_path`;
- referenciar também o crop de origem em `records.csv` por `source_crop_path` e `source_split`;
- salvar imagens geradas apenas em `generated/<generator>/`;
- salvar prompt e instruções usados em `prompts/<generator>/`;
- salvar máscara opcional em `masks/` quando houver edição localizada;
- se a anotação for feita em ferramenta externa como Roboflow, o export deve ser tratado como temporário; após a importação, a fonte de verdade volta a ser o pack sintético local;
- manter rastreabilidade de `pair_id`, gerador, modelo, prompt, severidade e tipo de anomalia em `records.csv`.

Pacote inicial oficial desta versão:

- `data/synthetic/tower_vision/v2026-04-16/anomaly_controlled_v1/`

Estado atual do pacote inicial:

- `20` imagens sintéticas aceitas para benchmark (`10` ChatGPT, `10` Gemini);
- máscaras importadas e internalizadas em `masks/val/` e `masks/test/`;
- `records.csv` preenchido com `mask_path` e `accepted_for_benchmark=true` para as amostras aprovadas;
- overlays de revisão em `reports/figures/tower_vision/v2026-04-16/anomaly_controlled_v1/mask_overlays/`;
- prancha única em `reports/figures/tower_vision/v2026-04-16/anomaly_controlled_v1/mask_overlays/contact_sheet.png`.
