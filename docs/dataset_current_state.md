# Dataset Current State

## 1. Visão geral

- Foram encontrados 300 arquivos de imagem e 1495 anotações em um pacote zip.
- Hipótese sobre o formato atual: O dataset atual parece ser um export COCO para detecção, com JSON único.

## 2. Estrutura atual

```text
imagens_torres_300/
├── annotations/ (1 arquivos)
└── images/ (300 arquivos)
    └── default/ (300 arquivos)
```

## 3. Inventário

- número de imagens: `300`
- número de anotações: `1495`
- número de arquivos de anotação: `1`
- extensões de imagem: `.jpg`
- classes encontradas: `torre`, `isoladores`

## 4. Formato das anotações

- formato aparente: `COCO`
- evidências curtas:
- arquivo de anotação `imagens_torres_300/annotations/instances_default.json`
- JSON com chaves `images`, `annotations` e `categories`
- anotações com campos `bbox`, `area`, `category_id` e `iscrowd`
- observações importantes: O JSON também contém `segmentation` e `attributes`, mas o uso primário aparente é detecção com bounding boxes.

## 5. Qualidade e consistência

- checks objetivos:
- imagens sem anotação: `0`
- anotações sem imagem: `0`
- arquivos vazios: `0`
- bboxes inválidas: `0`
- nomes incompatíveis JSON->imagens: `0`
- imagens extras fora do JSON: `0`
- duplicidade de annotation id: `0`
- duplicidade de file_name no JSON: `0`
- duplicidade de basename no pacote: `0`
- observações relevantes:
- dataset multiclasse: O dataset não é de classe única: há categorias `torre` e `isoladores`, o que exige filtro explícito se o benchmark futuro focar só em isoladores. Severidade aproximada: `média`.

## 6. Estatísticas básicas

- resoluções: min `5280x3956`, max `5280x3956`, média `5280.00x3956.00`
- largura: min `25.33`, max `654.60`, média `187.20`
- altura: min `34.77`, max `1712.95`, média `744.99`
- área relativa aproximada: min `0.000104`, max `0.049364`, média `0.012578`
- contagem por classe:
- `torre`: 591 anotações, área relativa média `0.031251`
- `isoladores`: 904 anotações, área relativa média `0.000370`

## 7. Agrupamentos úteis para split

- blocos temporais por minuto no nome: {'202509111050': 86, '202509111051': 85, '202509111049': 37, '202509111052': 86, '202509111053': 6}
- sequência contínua de captura no nome: 0001-0300
- limitações atuais:
- todas as imagens estão sob a mesma pasta de origem observável: imagens_torres_300/images/default
- a data aparente no nome é única para todo o conjunto: 20250911
- o prefixo de captura é único: DJI
- o sufixo no nome do arquivo é único: V

## 8. Conclusão

- O dataset está estruturalmente consistente e utilizável para diagnóstico, com formato COCO claro.
- próximos passos recomendados: Fixar a política de filtro por classe, decidir se `torre` entra no treino e definir group_ids baseados em blocos temporais ou sequência de captura antes de criar splits.

## Incertezas

- todas as imagens estão sob a mesma pasta de origem observável: imagens_torres_300/images/default
- a data aparente no nome é única para todo o conjunto: 20250911
- o prefixo de captura é único: DJI
- o sufixo no nome do arquivo é único: V