# Roteiro de apresentacao

## Pitch - 5 minutos

### Slide 1 - Problema e publico
Fala sugerida: Somos um desenvolvedor indie/publisher e precisamos entender quais caracteristicas aumentam a chance de um jogo se destacar na Steam, um mercado competitivo e com muita assimetria de popularidade.

### Slide 2 - Oportunidade
Fala sugerida: A base Steam Games Dataset permite combinar preco, reviews, owners estimados, peak CCU, categorias, generos, tags, plataformas e achievements para transformar historico de mercado em apoio a decisao.

### Slide 3 - Evidencias principais
Fala sugerida: O sucesso foi definido como top 30% de um score composto. As evidencias acionaveis indicam diferencas relevantes para multiplayer, modelo pago/gratuito e achievements.

### Slide 4 - Decisao proposta
Fala sugerida: Recomendamos priorizar features sociais quando fizer sentido ao genero, testar estrategia de preco com cuidado e planejar achievements conectados ao progresso real do jogo.

### Slide 5 - Impacto esperado
Fala sugerida: O impacto esperado e melhorar decisao de produto e posicionamento, usando metricas como reviews, peak CCU, donos estimados, tempo medio de jogo e taxa de sucesso do segmento.

## Apresentacao tecnica - 15 minutos

### Slide 6 - Base e preparacao
Mostrar origem Kaggle/Steam Games Dataset, quantidade de registros, tipos de variaveis e correcao do cabecalho `DiscountDLC count` apenas no carregamento.

### Slide 7 - Variavel-alvo
Explicar `success_commercial`: proxy top 30% do score composto com `log1p`, ranking percentual e media simples. Destacar ausencia de receita real.

### Slide 8 - Pre-processamento e features
Citar parsing de datas, owners, listas de tags/generos/categorias, plataformas, preco, achievements e remocao de colunas de vazamento.

### Slide 9 - EDA e inferencia
Mostrar distribuicao do alvo, preco, reviews, multiplayer, gratuito/pago e matriz de correlacao. Citar Mann-Whitney e qui-quadrado.

### Slide 10 - Modelos
Comparar Regressao Logistica, Arvore de Decisao, Random Forest, Extra Trees e HistGradientBoosting. XGBoost/LightGBM sao opcionais.

### Slide 11 - Metricas
Melhor modelo: `hist_gradient_boosting`. Accuracy=0.8285, Precision=0.6793, Recall=0.8115, F1=0.7395, ROC-AUC=0.9155.

### Slide 12 - Explicabilidade
Usar permutation importance. Variaveis mais relevantes: tag_count, game_age_years, has_trading_cards, price, discount. Traduzir como sinais de posicionamento, acabamento, preco e estrutura de tags.

### Slide 13 - Insights acionaveis
- Percebemos que jogos com multiplayer apresentaram taxa de sucesso de 44.9%, contra 26.7% nos demais jogos. Isso sugere que recursos sociais podem ampliar retencao, descoberta e recorrencia de uso. Por isso, recomendamos avaliar modos cooperativos, competitivos ou eventos online quando eles fizerem sentido para o genero. A decisao foi sustentada pela comparacao de taxa de sucesso e pelo teste estatistico registrado em `outputs/tables/statistical_tests.csv`. Apos 6 meses do lancamento, o sucesso pode ser avaliado por pico de usuarios simultaneos, reviews e retencao. A acao sera considerada bem-sucedida se o jogo superar em pelo menos 10% a taxa media de sucesso do seu segmento.
- Percebemos que jogos gratuitos tiveram taxa de sucesso de 17.6%, enquanto jogos pagos tiveram 33.4%. Isso sugere que preco zero pode aumentar alcance, mas precisa ser analisado junto ao modelo de monetizacao e ao genero. Por isso, recomendamos testar precos de entrada baixos, demos ou periodos promocionais sem depender automaticamente de free-to-play. A decisao foi sustentada pela comparacao entre grupos gratuito/pago. Apos 3 meses, o sucesso pode ser avaliado por conversao, reviews, donos estimados e receita por usuario. A acao sera considerada bem-sucedida se aumentar alcance sem reduzir a receita estimada por usuario abaixo da meta do projeto.
- Percebemos que jogos com achievements tiveram taxa de sucesso de 40.1%, contra 19.3% nos demais. Isso sugere que sistemas de progresso podem reforcar engajamento e sinalizar acabamento do produto. Por isso, recomendamos planejar achievements conectados a marcos reais de gameplay, evitando conquistas artificiais. A decisao foi sustentada pela comparacao entre grupos e pelas importancias do modelo, cujas variaveis mais relevantes incluem: tag_count, game_age_years, has_trading_cards, price, discount. Apos 3 meses, o sucesso pode ser avaliado por tempo medio de jogo e proporcao de jogadores que desbloqueiam conquistas intermediarias. A acao sera considerada bem-sucedida se elevar o tempo medio de jogo e a taxa de reviews positivas sem aumentar reclamacoes sobre repetitividade.

### Slide 14 - Limitacoes
Receita real ausente, proxy de sucesso, variaveis pos-lancamento, viés de popularidade, associacao nao causal e modelo como apoio, nao garantia.

### Slide 15 - Fechamento
Retomar a linha: problema -> oportunidade -> evidencias -> decisao -> impacto esperado.

## Possiveis perguntas da banca

- Por que nao usar apenas Estimated owners? Porque a coluna tem faixas largas e muitos empates; o score composto usa sinais complementares.
- Ha vazamento de dados? As colunas usadas diretamente no alvo e derivadas diretas foram removidas das features; variaveis pos-lancamento sao tratadas como explicativas.
- O modelo serve para jogos antes do lancamento? Parcialmente. Para pre-lancamento, use apenas features disponiveis antes do lancamento; o projeto atual explica e prediz com dados historicos de jogos publicados.
- Por que top 30%? E um corte operacional que cria uma classe positiva relevante e ainda com tamanho suficiente para treino e avaliacao.
- Correlação implica causalidade? Nao. Os resultados indicam associacoes e ajudam a priorizar hipoteses de decisao e experimentacao.
