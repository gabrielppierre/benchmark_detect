# Protocolo justo e defensável de benchmark para detectores de objetos em inspeção de torres

## Resumo executivo

Para o seu cenário, a recomendação mais defensável é: **usar early stopping para todos os modelos, de forma simétrica e pré-definida; monitorar uma métrica de detecção no conjunto de validação, não loss; validar a cada época; salvar o melhor checkpoint; usar `max_epochs = 100`; e escolher explicitamente a mesma regra de seleção de checkpoint para todas as famílias**. Em benchmark entre detectores heterogêneos, **loss não deve ser a métrica de seleção nem de ranking**, porque cada família otimiza objetivos diferentes e em escalas diferentes; já **`mAP50-95` é a opção metodologicamente mais segura como critério principal**, pois é a métrica padrão de detecção no estilo COCO, penaliza erros de localização e é bem mais alinhada ao objetivo real do detector do que loss bruta. citeturn32view0turn14view0turn35view3turn13view1turn24view1

A minha recomendação prática final é esta: **use `mAP50-95` de validação como monitor de early stopping e como critério de seleção do melhor checkpoint**, com **`mode = max`**, **`patience = 20`** e **`min_epochs = 25`**. Use **`mAP50-95` global no teste como ranking principal do benchmark**, mas mantenha um **ranking secundário orientado a `isoladores`**, reportando pelo menos **`AP50-95` da classe `isoladores`** e **`Recall` de `isoladores`**. Para a decisão prática do detector que alimentará a etapa de anomalia, eu **não escolheria pelo mAP global sozinho**: eu escolheria o modelo com melhor **`AP50-95` de `isoladores`**, desde que ele mantenha **recall de `isoladores` aceitável** e não colapse no ranking global. Isso preserva o rigor científico do benchmark multiclasse e, ao mesmo tempo, alinha a decisão de produto com a classe que realmente importa para a pipeline seguinte. citeturn13view1turn24view1turn22view0turn23view2

Também há uma decisão de justiça experimental que vale ser explicitada no relatório: **não deixe cada framework usar seu critério padrão de “melhor modelo” sem harmonização**. Na documentação da entity["company","Ultralytics","ai software company"], por exemplo, `best.pt` é salvo por um *fitness* que, por padrão, é `0.9 × mAP@0.5:0.95 + 0.1 × mAP@0.5`, enquanto o repositório oficial da entity["company","Megvii","computer vision company"] usa `AP50_95` para definir o melhor checkpoint. Se você não padronizar isso, o benchmark já nasce enviesado. citeturn14view0turn35view3

## Early stopping

**Early stopping é justo, sim, mas só sob condições estritas.** A literatura clássica trata early stopping como um mecanismo válido de regularização e seleção por validação, mas também mostra que a curva de validação é ruidosa e frequentemente apresenta múltiplos mínimos locais; por isso, o critério precisa ser definido antes dos experimentos e aplicado de forma idêntica a todos os modelos. Em outras palavras: early stopping **não é, por si só, uma fonte de viés**; o viés aparece quando cada família usa monitor diferente, frequência de validação diferente, *patience* diferente, ou quando o teste contamina a decisão de parada. citeturn32view0turn41view1turn41view2

No seu benchmark, a resposta metodologicamente mais forte é: **use early stopping em todos os quatro modelos**, mas **não use os defaults de cada biblioteca**. O que você deve padronizar é: conjunto de validação, frequência de validação, métrica monitorada, direção da métrica (`max`), regra de desempate e política de salvamento do checkpoint. Isso é mais justo do que treinar todos até a última época e comparar o último estado, porque o “último estado” é um artefato arbitrário de uma escolha de orçamento e não necessariamente o ponto de melhor generalização de cada família. citeturn32view0turn14view0turn35view3

**A métrica que eu recomendo monitorar é `mAP50-95` no conjunto de validação.** O motivo é simples: `mAP50` é útil, mas permissivo demais em localização; `recall` de `isoladores` é importante, mas ignora precisão e qualidade do encaixe da caixa; e loss de validação não é comparável entre famílias. Já `mAP50-95` captura precisão e recall ao longo do *ranking* de confiança e, além disso, penaliza caixas frouxas nos limiares mais altos de IoU. A própria documentação do Ultralytics descreve `mAP50-95` como uma visão mais abrangente do desempenho de detecção do que `mAP50`, e o trabalho de AP-Loss argumenta explicitamente que métricas do tipo AP são mais consistentes com a avaliação real de detecção do que métricas intermediárias de classificação. citeturn13view1turn24view1

Para o seu caso específico, em que `isoladores` é a classe mais importante, há duas formas defensáveis de proceder. A forma **mais padrão e publicável** é usar **`mAP50-95` global** para early stopping e seleção de checkpoint, deixando as métricas de `isoladores` para análise secundária e decisão operacional. A forma **mais orientada ao produto** é monitorar um **score composto pré-registrado**, por exemplo algo como `0,7 × AP50-95_isoladores + 0,3 × mAP50-95_global`; isso pode ser razoável, mas já é uma **inferência metodológica**, não um padrão consolidado da literatura, e portanto exige justificativa explícita no relatório e definição antes de rodar os experimentos. Se você quiser a opção mais conservadora e menos contestável em artigo, fique em `mAP50-95` global como monitor principal. citeturn22view0turn13view1turn24view1

**`patience` e `min_epochs` não têm um valor universal “correto” bem estabelecido na literatura para detecção em dataset pequeno**; aqui entramos em inferência metodológica. Como seu conjunto de validação tem só 43 imagens, a série temporal da métrica vai ser relativamente ruidosa. Isso torna `patience` muito baixo um risco claro de parada prematura. Por outro lado, com `max_epochs = 100`, usar `patience = 100` — como é o default do Ultralytics — praticamente desliga o early stopping. Por isso, para o seu cenário, eu recomendo começar com **`patience = 20`** e **`min_epochs = 25`**. Isso é lento o suficiente para não reagir a oscilações pequenas e, ao mesmo tempo, continua compatível com um orçamento de 100 épocas. Se quiser uma alternativa ligeiramente mais agressiva, `patience = 15` e `min_epochs = 20` ainda são defensáveis; abaixo disso, eu consideraria arriscado para um *val* tão pequeno. citeturn11view0turn32view0

A tabela abaixo resume a escolha do monitor de early stopping. Ela é uma **síntese metodológica** construída a partir das métricas padrão de detecção, do uso de AP/fórmulas de *fitness* nas implementações oficiais e do descompasso conhecido entre loss interna e qualidade final de detecção. citeturn13view1turn14view0turn35view3turn24view1turn26view0turn18search1

| Monitor | Vantagem principal | Problema principal | Recomendação |
|---|---|---|---|
| `val loss` | Fácil de acompanhar dentro de um mesmo modelo | Escala e significado mudam entre famílias; em alguns casos nem existe como caminho nativo de validação | **Não usar** para early stopping entre famílias |
| `mAP50` | Sinal costuma ser menos ruidoso; fácil de interpretar | Muito permissivo em localização | **Secundário**, não principal |
| `mAP50-95` | Padrão mais forte para detecção; penaliza localização ruim | Mais ruidoso que `mAP50` em *val* pequeno | **Melhor opção principal** |
| `Recall` de `isoladores` | Alinha com “não perder recortes” | Pode premiar muitos falsos positivos e caixas ruins | **Usar como restrição/critério secundário** |
| Score composto orientado a `isoladores` | Alinha benchmark com a pipeline futura | É customizado e menos padrão | **Opcional**, só se pré-registrado |

## Loss vs métricas de detecção

**Loss deve ser tratada como métrica de diagnóstico, não como critério de comparação entre famílias.** Em detectores diferentes, o que se chama de “loss” não é a mesma coisa. No Faster R-CNN da documentação oficial da entity["organization","PyTorch","deep learning framework"], o treinamento retorna perdas de classificação e regressão tanto do RPN quanto do detector; em modo de inferência, o modelo retorna previsões, não perdas. No RT-DETR da entity["company","Baidu","internet company"], a família DETR usa *bipartite matching* e perdas de caixa/classificação específicas desse paradigma; no Ultralytics, o melhor checkpoint é selecionado por um *fitness* baseado em AP; e no YOLOX o melhor checkpoint é atualizado a partir de `AP50_95`. Comparar “loss = 1,8” de um modelo com “loss = 0,9” de outro não tem interpretação metodológica limpa. citeturn18search1turn26view0turn14view0turn35view3

Isso vale ainda mais para **loss de validação**. No seu benchmark, ela é uma péssima candidata a critério central por três motivos. Primeiro, ela é **não comparável** entre arquiteturas. Segundo, em alguns frameworks ela é **não nativa** no fluxo de avaliação padronizado; no Faster R-CNN, por exemplo, o comportamento muda entre `train()` e `eval()`, o que torna a computação de perdas em validação algo adicional, não o mesmo pipeline de avaliação usado para AP. Terceiro, o que realmente interessa à sua aplicação são **caixas corretas e úteis para recorte de `isoladores`**, e não a minimização de uma função surrogata específica de cada família. citeturn10view7turn18search1turn24view1

O que acompanhar por época, então? Eu sugiro separar em duas camadas. A camada de **diagnóstico de treino** deve incluir: `train loss` total e, quando a implementação expuser, perdas por componente; *learning rate*; tempo por época; e eventualmente `val loss` apenas para inspeção interna do mesmo modelo. A camada de **seleção e comparação** deve incluir: `mAP50`, `mAP50-95`, `precision`, `recall`, `AP50` e `AP50-95` por classe, além de curvas por época para as métricas principais e curvas PR por classe no melhor checkpoint. A documentação do Ultralytics expõe exatamente esse tipo de detalhamento, incluindo métricas por classe e curvas como F1/PR. citeturn13view1turn13view2turn14view3

A consequência prática é objetiva: **acompanhe loss, mas não escolha modelo por loss; acompanhe precision/recall, mas não ranqueie benchmark só por precision/recall; escolha checkpoint e faça ranking por métricas de detecção baseadas em AP, preferencialmente `mAP50-95` no geral e `AP50-95` por classe quando a decisão for orientada à aplicação**. citeturn13view1turn24view1

## Protocolo recomendado para o meu benchmark

O seu cenário já tem dois acertos metodológicos importantes: **split congelado** e **agrupamento temporal para reduzir vazamento**. Isso é especialmente valioso em inspeção industrial, porque evita que imagens quase duplicadas ou fortemente correlacionadas atravessem os subconjuntos e inflacionem o resultado. A partir daqui, a principal prioridade é garantir que **todo o processo de seleção de modelo aconteça só no conjunto de validação**, deixando o teste estritamente para a avaliação final. citeturn40view1turn23view2

Quanto à **inicialização**, eu recomendo fortemente **fine-tuning com pesos pré-treinados em COCO para todos os modelos**. O repositório oficial do YOLOX recomenda explicitamente começar de pesos pré-treinados em COCO para custom data; o tutorial oficial de detecção do TorchVision usa fine-tuning de modelo pré-treinado justamente porque o dataset é pequeno; o RT-DETR oferece pesos COCO e também versões mais fortes pré-treinadas em Objects365; e os exemplos do YOLO11 começam de pesos pré-treinados. Para benchmark justo, isso implica uma regra simples: **use a inicialização COCO-pretrained para os quatro modelos e não use a variante RT-DETR-R18 com pré-treino adicional em Objects365**, a menos que você consiga equiparar esse ganho extra de pré-treino para todas as demais famílias. citeturn15view1turn17view0turn17view1turn10view2turn36view3

Quanto à **resolução**, a decisão justa é usar a **mesma resolução efetiva na trilha principal do benchmark**. Como YOLO11, YOLOX e RT-DETR têm receitas e tabelas oficiais muito centradas em `640`, a solução mais limpa para o leaderboard principal é **usar 640 para todos**. Isso não significa que 640 seja necessariamente a melhor resolução para o seu problema — sobretudo com imagens originalmente de alta resolução e `isoladores` potencialmente pequenos —, mas significa que a comparação principal fica menos contaminada por um confounder fortíssimo. Se, depois, você quiser avaliar a melhor configuração operacional para produção, faça uma **segunda trilha pré-registrada** em resolução maior, ainda comum a todos, como 960 ou 1024. O que não é bom metodologicamente é deixar cada família “escolher seu tamanho ideal” dentro do mesmo benchmark principal. citeturn11view0turn16search0turn36view3turn10view2

Quanto a **augmentations**, eu não recomendaria tentar forçar equivalência literal entre pipelines complexos e nativos de frameworks distintos. A literatura de benchmarking e o trabalho do detrex deixam claro que diferenças de implementação e hiperparâmetros têm impacto real e dificultam comparações justas. Por isso, para o **benchmark principal**, eu sugiro usar uma **política comum e leve**, disponível em todas as famílias: *flip* horizontal se fizer sentido físico, pequenas variações de escala, e jitter fotométrico moderado. Eu evitaria, no protocolo central, misturar estratégias muito específicas como mosaic/mixup/copy-paste apenas em algumas famílias, porque isso troca “comparação de modelos” por “comparação de modelos mais receitas de dados diferentes”. Se você quiser, pode deixar um **benchmark secundário nativo-da-família** em apêndice. citeturn22view1turn22view0turn36view0

No que diz respeito ao **orçamento de treino**, **`max_epochs = 100` para todos, validação a cada época e salvamento do melhor checkpoint** é uma escolha bastante defensável para o seu caso. Ela conversa bem com os exemplos de treino de YOLO11 e RT-DETR, corrige o `eval_interval = 10` do YOLOX — que é inadequado se você quer fazer model selection fino em dataset pequeno — e continua barata o bastante para permitir rodadas múltiplas com sementes diferentes. O ponto mais importante aqui é: **100 deve ser o teto comum, não a época “de verdade” do modelo**; o modelo “de verdade” é o melhor checkpoint de validação dentro desse teto. citeturn10view2turn10view6turn16search0turn35view2turn35view3

Por fim, para um benchmark que você quer defender em relatório técnico ou artigo, eu faria **três sementes no mínimo e cinco se o custo permitir**, usando a mesma lista de sementes para todos os modelos. A literatura de reprodutibilidade e variância mostra que uma única execução pode produzir conclusões frágeis; isso fica ainda mais importante em dataset pequeno. O ideal é: para cada semente, treine com o mesmo protocolo, selecione o melhor checkpoint por validação, avalie no teste e, no relatório, publique média e desvio-padrão — ou, melhor ainda, intervalo de confiança por bootstrap quando possível. citeturn40view1turn23view2turn23view3

## Critério de ranking dos modelos

Para manter o benchmark intelectualmente limpo, eu recomendo um esquema de **dupla leitura**: um **ranking geral do benchmark** e um **ranking orientado à classe crítica**. O **ranking geral** deve usar como métrica principal **`mAP50-95` no conjunto de teste**, agregado ao longo das sementes. Essa é a métrica que melhor representa o desempenho multiclasse do detector de forma padronizada e que mais facilmente será entendida por revisores, gestores técnicos e leitores externos. `mAP50` pode aparecer junto, mas como apoio. citeturn13view1turn35view3turn14view0

O **ranking orientado a `isoladores`** deve usar como métrica principal **`AP50-95` da classe `isoladores`**. Aqui eu prefiro `AP` de classe a `recall` puro porque a sua etapa futura depende não apenas de “achar algo”, mas de **gerar recortes úteis** para uma etapa posterior. `AP50-95` preserva a informação sobre *precision-recall* e ainda pune caixas ruins, o que é relevante para a qualidade dos recortes. O `Recall` de `isoladores` deve entrar como métrica de salvaguarda, não como métrica soberana. citeturn13view1turn24view1

A resposta objetiva para a sua pergunta “**faz sentido selecionar o melhor modelo por desempenho em `isoladores`?**” é: **sim, para a decisão prática de pipeline; não, como única manchete do benchmark multiclasse**. Se você publicar ou documentar só um leaderboard baseado em `isoladores`, você enfraquece a defesa de que está comparando detectores de objetos em um problema multiclasse. Mas, se você publicar **o ranking geral por `mAP50-95`** e, adicionalmente, **um ranking orientado a `isoladores`**, você preserva a integridade do benchmark e ainda toma a decisão de engenharia correta para a etapa de anomalia. citeturn22view0turn13view1

Na prática, eu sugiro a seguinte regra de decisão para o detector que vai alimentar a etapa não supervisionada de anomalias: **defina antecipadamente, no conjunto de validação, um piso operacional de `Recall` de `isoladores`; entre os modelos que satisfizerem esse piso, escolha o maior `AP50-95` de `isoladores`; use `mAP50-95` global como desempate e faça uma inspeção qualitativa das caixas geradas**. Essa regra é mais alinhada à sua pipeline do que simplesmente pegar o campeão global de mAP. O ponto crítico é que o piso de recall deve ser **pré-definido antes de olhar o teste**, para não transformar o teste em conjunto de tuning. citeturn13view1turn23view2turn40view1

## Riscos metodológicos

O primeiro grande risco é **comparar modelos com pré-treinos diferentes**. No RT-DETR oficial, há variantes com ganho adicional por Objects365; se você comparar isso com YOLO11s, YOLOX-s e Faster R-CNN apenas COCO-pretrained, você estará misturando “arquitetura” com “vantagem de pré-treino”. Para artigo e relatório técnico, isso é facilmente questionável. A solução é simples: alinhar todos em COCO-pretrained na trilha principal. citeturn36view3turn15view1turn17view1

O segundo risco é **deixar defaults de framework determinarem a seleção do melhor checkpoint**. O exemplo mais evidente é o Ultralytics, cujo `best.pt` usa um *fitness* próprio, enquanto o YOLOX usa `AP50_95`. Se você não sobrescrever isso, não estará usando o mesmo critério de model selection. É um viés silencioso, porém importante. citeturn14view0turn35view3

O terceiro risco é **superestimar resultado por tuning implícito de hiperparâmetros e receitas**. Há evidência de que comparações só são realmente justas quando o objetivo de tuning é o mesmo e quando o custo de tuning também entra na conta; além disso, frameworks independentes e implementações distintas aumentam o risco de comparação desigual. O trabalho do detrex foi motivado justamente pela dificuldade de obter comparações justas quando famílias diferentes vivem em código e receitas independentes. No seu caso, isso sugere duas boas práticas: reduzir o tuning livre no benchmark principal e documentar integralmente os hiperparâmetros usados. citeturn22view0turn22view1turn22view2

O quarto risco é **tirar conclusão com execução única**. Em dataset pequeno, a combinação de inicialização aleatória, ordem dos lotes, augmentations estocásticas e detalhes de framework pode mudar o resultado o suficiente para inverter o vencedor. A literatura recente de reprodutibilidade recomenda publicar sementes, controlar a aleatoriedade tanto quanto possível e avaliar múltiplas execuções para reportar variância. citeturn40view1turn23view2

O quinto risco é **usar o conjunto de teste para qualquer decisão intermediária**: escolher época, escolher limiar de confiança, escolher NMS, escolher resolução, escolher resolução por classe ou decidir qual detector vai para a etapa de anomalia. Tudo isso deve ser decidido em validação. O teste precisa continuar sendo o bloco final de auditoria do protocolo. citeturn32view0turn40view1

O sexto risco é **confundir justiça experimental com igualdade mecânica absoluta**. “Mesma quantidade de épocas” não significa “mesmo custo computacional”, e “mesmas augmentations exatas” nem sempre significa “mesma informação efetiva” entre frameworks. Por isso, no relatório, vale a pena declarar explicitamente qual noção de justiça você está adotando. Para o seu contexto, a noção mais útil é: **mesmo split, mesmo pré-treino de base, mesma resolução principal, mesma política comum de validação, mesmo orçamento máximo de épocas, mesmo monitor de seleção e mesma regra de checkpoint**; e, em paralelo, **reportar tempo de treino e velocidade de inferência como desfechos adicionais**, não como parte escondida do protocolo. citeturn22view0turn22view1turn26view0

## Protocolo final sugerido

Abaixo está o bloco objetivo que eu implementaria como protocolo principal do benchmark.

- **Split**: manter exatamente o split congelado atual; `train` para atualização de pesos, `val` para early stopping, seleção de checkpoint, escolha de limiares operacionais e qualquer decisão de protocolo; `test` apenas para avaliação final. citeturn32view0turn40view1

- **Inicialização**: usar **pesos COCO-pretrained** para **YOLO11s**, **YOLOX-s**, **Faster R-CNN ResNet50-FPN v2** e **RT-DETR-R18**. **Não usar** RT-DETR-R18 com pré-treino adicional em Objects365 na trilha principal. citeturn10view2turn15view1turn17view1turn36view3

- **Resolução principal**: usar **640** para todos os modelos no benchmark principal. Se o desempenho em `isoladores` sugerir limitação por escala, abrir depois uma **segunda trilha comum** em resolução maior, mas sem misturar resoluções no mesmo leaderboard. citeturn11view0turn16search0turn36view3turn10view2

- **Augmentations**: usar uma política **leve e comum** no benchmark principal, evitando family-specific extras difíceis de igualar entre frameworks. Guardar um experimento secundário “receita nativa da família” apenas como análise complementar. citeturn22view1turn22view0

- **`max_epochs`**: **100** para todos. citeturn10view2turn10view6turn11view0

- **`validate_every`**: **1 época** para todos. No YOLOX, isso deve ser explicitamente sobrescrito, porque a configuração base usa outro intervalo de avaliação. citeturn16search0turn35view2

- **`save_best`**: **sim**. **`save_last`**: **sim**. Armazenar ambos. O checkpoint “best” deve obedecer à mesma métrica para todas as famílias. citeturn14view0turn35view3

- **Early stopping**: **ativado para todos**. citeturn32view0turn11view0

- **Monitor**: **`mAP50-95` no conjunto de validação**. citeturn13view1turn24view1

- **`mode`**: **`max`**. citeturn14view0turn35view3

- **`patience`**: **20** como valor inicial recomendável. citeturn32view0turn11view0

- **`min_epochs`**: **25** como valor inicial recomendável antes de permitir parada. Isso não vem como recomendação universal das fontes; é uma inferência metodológica para amortecer ruído de validação em dataset pequeno. citeturn32view0

- **Métricas por época**: `train loss` total e componentes quando disponíveis, `learning rate`, `mAP50`, `mAP50-95`, `precision`, `recall`, `AP50` e `AP50-95` por classe, `precision` e `recall` por classe, e curvas por época das métricas principais. `val loss`, se calculada, fica só para diagnóstico dentro do mesmo modelo. citeturn13view1turn14view3turn18search1turn26view0

- **Ranking final do benchmark**: **média do `mAP50-95` no teste ao longo de 3 sementes**, com desvio-padrão ou intervalo de confiança. Se puder pagar o custo, subir para 5 sementes. citeturn40view1turn23view2

- **Ranking orientado a `isoladores`**: reportar **`AP50-95_isoladores`** e **`Recall_isoladores`** no teste, também agregados por semente. citeturn13view1turn24view1

- **Escolha prática do detector para a etapa de anomalia**: decidir em **validação**, não em teste, usando a regra “maior `AP50-95_isoladores` entre os modelos que passam em um piso de `Recall_isoladores` previamente definido”; usar `mAP50-95` global como desempate. citeturn13view1turn40view1

- **Reprodutibilidade**: publicar lista de sementes, versões de bibliotecas, *commit hash* do código, GPU, resolução, política de augmentations, hiperparâmetros de treino, regra de seleção do checkpoint e logs por época. citeturn40view1turn23view2

Em estilo direto, a recomendação final fica assim: **faça X**: use early stopping simétrico para todos; monitore `mAP50-95` de validação; valide a cada época; salve o melhor checkpoint; use `max_epochs = 100`, `patience = 20` e `min_epochs = 25`; use pesos COCO-pretrained para todos; publique também métricas por classe e múltiplas sementes. **Não faça Y**: não use loss para ranquear famílias; não deixe cada framework escolher seu próprio critério de `best checkpoint`; não misture pré-treinos desbalanceados; não use o teste para escolher época, limiar ou detector. **Use esta métrica para early stopping**: `mAP50-95` de validação. **Use esta métrica para ranking final**: `mAP50-95` global no teste. **Use esta métrica para a decisão de pipeline**: `AP50-95` de `isoladores`, sob restrição mínima de `Recall_isoladores`. citeturn14view0turn35view3turn13view1turn24view1turn40view1

## Referências

- **Lutz Prechelt, “Early Stopping — But When?”** Base clássica para justificar early stopping como regularização/seleção por validação e para discutir o trade-off entre critério rápido e critério lento; também sustenta a ideia de salvar o melhor ponto de validação, não o último. citeturn32view0

- **Documentação de configuração de treino do Ultralytics**. Sustenta `epochs`, `patience`, `imgsz`, `save`, `save_period` e o fato de que a biblioteca usa early stopping baseado em métricas de validação. citeturn11view0turn11view3

- **Guia “Customizing Trainer” do Ultralytics**. Sustenta que o `best.pt` é salvo por um *fitness* padrão (`0.9 × mAP50-95 + 0.1 × mAP50`) e que esse comportamento pode e deve ser sobrescrito quando você quer harmonizar o benchmark por outra métrica. citeturn14view0turn14view1

- **Guia de métricas do Ultralytics**. Sustenta o significado de `precision`, `recall`, `mAP50`, `mAP50-95`, métricas por classe e a distinção entre métricas globais e métricas específicas por classe. citeturn13view1turn13view2

- **Documentação e código oficial do YOLOX para custom data**. Sustentam o uso de pesos COCO-pretrained, os defaults de `input_size = 640`, `max_epoch = 300`, `eval_interval = 10` e o fato de que o melhor checkpoint é atualizado por `AP50_95`. citeturn15view1turn16search0turn35view3

- **Documentação do TorchVision / Faster R-CNN e tutorial oficial de fine-tuning**. Sustentam que o Faster R-CNN retorna perdas em treino e previsões em inferência, além da recomendação prática de fine-tuning em dataset pequeno a partir de pesos pré-treinados. citeturn18search1turn17view0turn17view1

- **Artigo e repositório oficial do RT-DETR**. Sustentam a natureza end-to-end da família DETR, o uso de *bipartite matching*, a centralidade de AP no benchmark, os resultados em 640 e a disponibilidade de variantes com COCO e com Objects365 — ponto crucial para justiça de pré-treino. citeturn26view0turn36view3turn10view6

- **Chen et al., “Towards Accurate One-Stage Object Detection with AP-Loss”**. Sustenta a ideia de que métricas do tipo AP são mais consistentes com a avaliação final da detecção do que objetivos intermediários de classificação, reforçando a preferência por AP/mAP em seleção e comparação. citeturn24view1

- **Ren et al., “detrex: Benchmarking Detection Transformers”**. Sustenta que benchmarks entre famílias/implementações independentes são difíceis de tornar justos e que hiperparâmetros e código-base impactam materialmente a comparação. citeturn22view1

- **Turner et al., “Benchmarking Neural Network Training Algorithms”** e Sivaprasad et al., “Optimizer Benchmarking Needs to Account for Hyperparameter Tuning”**. Sustentam que comparação justa exige mesmo objetivo de tuning e cuidado com o custo de ajuste de hiperparâmetros. citeturn22view0turn22view2

- **Henderson et al., “Deep Reinforcement Learning that Matters”** e Semmelrock et al., “Reproducibility in machine-learning-based research”**. Embora não sejam específicos de detecção, sustentam diretamente a necessidade de múltiplas sementes, variância, testes de significância e documentação completa para comparações confiáveis. citeturn23view2turn40view1