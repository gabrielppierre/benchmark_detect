# Synthetic Visual Review


## Escopo
- Revisao visual das 20 imagens sinteticas do pack `anomaly_controlled_v1`.
- Objetivo: avaliar plausibilidade visual e coerencia do `severity` original (`moderate`).
- O `severity` original em `records.csv` foi preservado; a sugestao desta revisao foi registrada em `notes`.

## Conclusao curta
- As 20 imagens sao candidatas validas para experimento controlado.
- O principal ajuste recomendado e recalibrar severidade: varias saidas do Gemini ficaram acima de `moderate`.
- As amostras de `severe contamination` sao visualmente plausiveis, mas devem ser tratadas como `severe` ou renomeadas para `contamination`.

## Sugestoes por imagem

| record_id | generator | anomaly_type | severity_original | severity_sugerido | valida | observacao |
| --- | --- | --- | --- | --- | --- | --- |
| test_1_DJI_20250911105233_0257_V__1278__chatgpt | chatgpt | crack | moderate | moderate | true | plausible_localized_crack |
| test_1_DJI_20250911105233_0257_V__1278__gemini | gemini | crack | moderate | moderate_to_severe | true | plausible_crack_stronger_than_target |
| test_2_DJI_20250911105238_0263_V__1309__chatgpt | chatgpt | partial chipping | moderate | moderate | true | plausible_localized_chip |
| test_2_DJI_20250911105238_0263_V__1309__gemini | gemini | partial chipping | moderate | moderate_to_severe | true | chip_larger_than_target_but_plausible |
| test_3_DJI_20250911105247_0276_V__1374__chatgpt | chatgpt | burn mark | moderate | moderate | true | plausible_burn_mark |
| test_3_DJI_20250911105247_0276_V__1374__gemini | gemini | burn mark | moderate | moderate_to_severe | true | burn_mark_stronger_than_target |
| test_4_DJI_20250911105257_0291_V__1450__chatgpt | chatgpt | severe contamination | moderate | severe | true | contamination_plausible_but_stronger_than_target |
| test_4_DJI_20250911105257_0291_V__1450__gemini | gemini | severe contamination | moderate | severe | true | contamination_plausible_but_stronger_than_target |
| test_5_DJI_20250911105303_0300_V__1491__chatgpt | chatgpt | localized surface damage | moderate | moderate | true | plausible_localized_surface_damage |
| test_5_DJI_20250911105303_0300_V__1491__gemini | gemini | localized surface damage | moderate | moderate_to_severe | true | surface_damage_stronger_than_target |
| val_1_DJI_20250911105200_0210_V__1042__chatgpt | chatgpt | crack | moderate | moderate | true | plausible_localized_crack |
| val_1_DJI_20250911105200_0210_V__1042__gemini | gemini | crack | moderate | moderate_to_severe | true | crack_cluster_stronger_than_target |
| val_2_DJI_20250911105207_0219_V__1087__chatgpt | chatgpt | partial chipping | moderate | moderate | true | plausible_chip_near_upper_bound_of_moderate |
| val_2_DJI_20250911105207_0219_V__1087__gemini | gemini | partial chipping | moderate | moderate_to_severe | true | chip_larger_than_target_but_plausible |
| val_3_DJI_20250911105214_0229_V__1138__chatgpt | chatgpt | burn mark | moderate | moderate | true | plausible_burn_mark |
| val_3_DJI_20250911105214_0229_V__1138__gemini | gemini | burn mark | moderate | moderate_to_severe | true | burn_mark_stronger_than_target |
| val_4_DJI_20250911105221_0240_V__1193__chatgpt | chatgpt | severe contamination | moderate | severe | true | contamination_plausible_but_stronger_than_target |
| val_4_DJI_20250911105221_0240_V__1193__gemini | gemini | severe contamination | moderate | severe | true | contamination_plausible_but_stronger_than_target |
| val_5_DJI_20250911105226_0247_V__1225__chatgpt | chatgpt | localized surface damage | moderate | moderate_to_severe | true | localized_damage_stronger_than_target |
| val_5_DJI_20250911105226_0247_V__1225__gemini | gemini | localized surface damage | moderate | moderate | true | plausible_localized_surface_damage |

## Observacoes metodologicas
- `severity_original=moderate` continua sendo a intencao do prompt, nao o julgamento final da revisao.
- `review_suggested_severity` em `notes` serve como apoio para relabeling humano posterior.
- `accepted_for_benchmark` permanece `false` ate conclusao da anotacao de mascara e curadoria final.
