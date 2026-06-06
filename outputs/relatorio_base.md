# Relatorio-base - Predicao de sucesso comercial de jogos Steam

## Aplicacao

O problema de negocio e apoiar um desenvolvedor indie ou publisher a entender quais caracteristicas estao associadas a melhor desempenho comercial na Steam. O projeto usa dados historicos de jogos para predizer e explicar `success_commercial`.

## Base de Dados

A base principal e `dataset/games.csv`, do Steam Games Dataset. O pipeline processou 122,611 registros e 49 colunas apos a engenharia de atributos. Durante a leitura, o cabecalho `DiscountDLC count` e tratado como duas colunas: `Discount` e `DLC count`, porque as linhas possuem 40 campos. O CSV bruto nao e editado. O feedback do monitor foi incorporado pela definicao explicita da variavel-alvo.

As principais variaveis originais usadas na analise incluem preco, data de lancamento, plataformas, estimated owners, reviews positivas/negativas, peak CCU, achievements, metacritic, categorias, generos e tags. A preparacao inclui conversao numerica, parsing de datas, parsing de faixas de owners, tratamento de nulos por imputacao nos modelos, transformacao logaritmica de variaveis assimetricas e criacao de indicadores binarios para plataformas e caracteristicas de gameplay.

## Estatistica Descritiva e Inferencia

Foram geradas distribuicoes de preco, reviews, score comercial, grupos multiplayer/gratuito/achievements e matriz de correlacao. As visualizacoes respondem perguntas de negocio: tamanho do grupo de sucesso, diferenca entre jogos pagos e gratuitos, associacao de multiplayer com sucesso, concentracao de reviews e relacoes entre variaveis numericas. Os testes estatisticos principais foram:

- Mann-Whitney U: Score comercial: jogos multiplayer vs nao multiplayer (p-valor=<1e-300)
- Mann-Whitney U: Score comercial: jogos gratuitos vs pagos (p-valor=<1e-300)
- Mann-Whitney U: Score comercial: jogos indie vs nao indie (p-valor=<1e-300)
- Qui-quadrado: Sucesso comercial vs multiplayer (p-valor=<1e-300)
- Qui-quadrado: Sucesso comercial vs gratuito (p-valor=<1e-300)

## Metodos Avaliados

Foram avaliados uma Regressao Logistica como baseline, uma Arvore de Decisao simples, Random Forest, Extra Trees e HistGradientBoosting como modelos avancados. XGBoost e LightGBM sao treinados apenas se estiverem instalados no ambiente.

## Metricas de Avaliacao

As metricas usadas foram accuracy, precision, recall, F1-score, ROC-AUC e matriz de confusao. O modelo vencedor foi `hist_gradient_boosting`.

## Metodos de Avaliacao

A avaliacao usa divisao treino/teste estratificada, com teste de 20% da base, e validacao cruzada estratificada com 3 folds no conjunto de treino.

## Resultados e Discussao

Metricas no conjunto de teste:

| model | accuracy | precision | recall | f1 | roc_auc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | 0.8285 | 0.6793 | 0.8115 | 0.7395 | 0.9155 | 14347 | 2819 | 1387 | 5970 |
| random_forest | 0.8307 | 0.6945 | 0.7780 | 0.7339 | 0.9117 | 14648 | 2518 | 1633 | 5724 |
| extra_trees | 0.8143 | 0.6613 | 0.7809 | 0.7162 | 0.9008 | 14224 | 2942 | 1612 | 5745 |
| logistic_regression_baseline | 0.8008 | 0.6381 | 0.7763 | 0.7004 | 0.8862 | 13927 | 3239 | 1646 | 5711 |
| decision_tree_baseline | 0.7949 | 0.6257 | 0.7877 | 0.6974 | 0.8880 | 13699 | 3467 | 1562 | 5795 |

Metricas de validacao cruzada:

| model | cv_f1_mean | cv_f1_std | cv_roc_auc_mean | cv_roc_auc_std | cv_accuracy_mean | cv_accuracy_std |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_baseline | 0.7006 | 0.0027 | 0.8867 | 0.0012 | 0.8011 | 0.0016 |
| decision_tree_baseline | 0.6964 | 0.0011 | 0.8860 | 0.0018 | 0.7965 | 0.0059 |
| random_forest | 0.7356 | 0.0016 | 0.9098 | 0.0007 | 0.8318 | 0.0013 |
| extra_trees | 0.7205 | 0.0011 | 0.9017 | 0.0008 | 0.8171 | 0.0004 |
| hist_gradient_boosting | 0.7389 | 0.0013 | 0.9144 | 0.0005 | 0.8280 | 0.0010 |

A variavel-alvo foi definida como top 30% do score composto. O corte calculado foi `0.524124` e a taxa positiva foi 30.0%.

## Explicabilidade

Como o melhor modelo atual e baseado em HistGradientBoosting, a explicabilidade usa permutation importance. As variaveis mais relevantes no pipeline final foram: tag_count, game_age_years, has_trading_cards, price, discount. Em linguagem de negocio, isso indica que variedade/quantidade de tags, idade do jogo, sinais de acabamento/recursos da Steam, preco e aprovacao observada em EDA ajudam a explicar diferencas de desempenho na base.

## Limitacoes e cuidados de interpretacao

- A base nao possui receita real por jogo; `success_commercial` e uma proxy, nao faturamento observado.
- O score composto combina sinais de alcance e engajamento, mas nao substitui informacoes financeiras reais.
- Algumas variaveis analisadas, como reviews e Metacritic, sao pos-lancamento; elas ajudam a explicar jogos ja publicados, mas nao devem ser tratadas como dados garantidos antes do lancamento.
- Jogos muito populares podem influenciar padroes por viés de popularidade e exposicao da plataforma.
- Os testes indicam associacao estatistica, mas correlacao nao implica causalidade.
- O modelo apoia decisoes de produto, marketing e posicionamento, mas nao garante sucesso comercial futuro.

## Conclusao

O projeto transforma o problema de sucesso comercial em uma classificacao interpretavel, baseada em sinais complementares de alcance, reviews e engajamento. Os resultados devem ser usados como apoio a decisao, nao como garantia de desempenho futuro. Trabalhos futuros podem estimar receita com hipoteses de preco, separar modelos pre-lancamento e pos-lancamento, testar calibracao de probabilidades e construir um dashboard para explorar generos especificos.

## Insights acionaveis

- Percebemos que jogos com multiplayer apresentaram taxa de sucesso de 44.9%, contra 26.7% nos demais jogos. Isso sugere que recursos sociais podem ampliar retencao, descoberta e recorrencia de uso. Por isso, recomendamos avaliar modos cooperativos, competitivos ou eventos online quando eles fizerem sentido para o genero. A decisao foi sustentada pela comparacao de taxa de sucesso e pelo teste estatistico registrado em `outputs/tables/statistical_tests.csv`. Apos 6 meses do lancamento, o sucesso pode ser avaliado por pico de usuarios simultaneos, reviews e retencao. A acao sera considerada bem-sucedida se o jogo superar em pelo menos 10% a taxa media de sucesso do seu segmento.
- Percebemos que jogos gratuitos tiveram taxa de sucesso de 17.6%, enquanto jogos pagos tiveram 33.4%. Isso sugere que preco zero pode aumentar alcance, mas precisa ser analisado junto ao modelo de monetizacao e ao genero. Por isso, recomendamos testar precos de entrada baixos, demos ou periodos promocionais sem depender automaticamente de free-to-play. A decisao foi sustentada pela comparacao entre grupos gratuito/pago. Apos 3 meses, o sucesso pode ser avaliado por conversao, reviews, donos estimados e receita por usuario. A acao sera considerada bem-sucedida se aumentar alcance sem reduzir a receita estimada por usuario abaixo da meta do projeto.
- Percebemos que jogos com achievements tiveram taxa de sucesso de 40.1%, contra 19.3% nos demais. Isso sugere que sistemas de progresso podem reforcar engajamento e sinalizar acabamento do produto. Por isso, recomendamos planejar achievements conectados a marcos reais de gameplay, evitando conquistas artificiais. A decisao foi sustentada pela comparacao entre grupos e pelas importancias do modelo, cujas variaveis mais relevantes incluem: tag_count, game_age_years, has_trading_cards, price, discount. Apos 3 meses, o sucesso pode ser avaliado por tempo medio de jogo e proporcao de jogadores que desbloqueiam conquistas intermediarias. A acao sera considerada bem-sucedida se elevar o tempo medio de jogo e a taxa de reviews positivas sem aumentar reclamacoes sobre repetitividade.

## Bibliografia/Referencias

- Steam Games Dataset, Kaggle.
- Especificacao do Projeto Final de Ciencia de Dados, UFAL, 2026.
- Scikit-learn documentation.
- SciPy documentation.
