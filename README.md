# Steam Games Market Insights

Projeto final de Ciência de Dados sobre jogos da Steam.

A ideia é simples: usar dados públicos de jogos já publicados para entender quais características aparecem associadas a um maior sucesso comercial estimado. O foco não é prever receita real, mas apoiar decisões de produto, preço e posicionamento para desenvolvedores indie e publishers.

## Problema

Um desenvolvedor ou publisher pode ter várias dúvidas antes de investir em um jogo:

- vale apostar em multiplayer?
- free-to-play parece uma boa estratégia?
- achievements e recursos da Steam fazem diferença?
- quais sinais aparecem com mais força nos jogos que performaram melhor?

Este projeto tenta responder esse tipo de pergunta com dados, modelos e análises estatísticas.

## Dataset

A base usada é o Steam Games Dataset, disponível no Kaggle:

https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?resource=download

O pipeline usa principalmente:

text dataset/games.csv

O arquivo dataset/games.json pode ficar na pasta como apoio, mas o CSV é a fonte principal.

Como a pasta dataset/ está no .gitignore, os dados não são enviados para o GitHub. Para rodar o projeto, baixe a base no Kaggle e deixe os arquivos assim:

text dataset/   games.csv   games.json

Durante o carregamento, o cabeçalho DiscountDLC count é tratado como duas colunas: Discount e DLC count. Essa correção acontece só no código de leitura; o CSV bruto não é alterado.

## Variável-alvo

A variável que queremos prever é:

text success_commercial

Ela foi criada no projeto e representa uma proxy binária de sucesso comercial estimado.

Isso significa duas coisas:

1. É uma aproximação, porque a base não tem receita real dos jogos.
2. É binária, porque cada jogo fica em uma de duas classes:
   - 1: sucesso comercial estimado;
   - 0: não sucesso comercial estimado.

A classe positiva é formada pelos jogos no top 30% de um score composto por:

- ponto médio de Estimated owners;
- total de reviews;
- pico de jogadores simultâneos;
- recomendações;
- tempo médio de jogo.

Antes de compor o score, as métricas passam por log1p e ranking percentual, para reduzir o peso de valores extremos.

As colunas usadas para construir o alvo, e suas derivadas diretas, são removidas das features de treino para evitar vazamento de dados. A explicação completa está em:

text outputs/definicao_sucesso_comercial.md

## Estrutura do projeto

text src/                         código principal do pipeline app/                         aplicação Streamlit dataset/                     dados locais, não versionados outputs/figures/             gráficos gerados outputs/tables/              tabelas, testes, métricas e importâncias outputs/models/              modelo salvo localmente outputs/app_reports/         saídas geradas pelo app Streamlit outputs/relatorio_base.md    base textual do relatório final outputs/insights_acionaveis.md outputs/roteiro_apresentacao.md

## Instalação

bash python3 -m venv .venv source .venv/bin/activate python3 -m pip install -r requirements.txt

No Windows, a ativação do ambiente virtual costuma ser:

bash .venv\Scripts\activate

## Rodando o pipeline principal

Com o ambiente ativado e o dataset na pasta correta:

bash python3 -m src.main

Esse comando gera os principais arquivos em outputs/, incluindo métricas, tabelas, figuras, relatório-base e insights.

## Aplicação Streamlit

Também há uma aplicação local em Streamlit para explorar o projeto de forma interativa.

Ela permite:

- carregar games.csv ou um CSV compatível;
- selecionar modelos;
- treinar e comparar classificadores;
- visualizar métricas e gráficos;
- gerar artefatos em outputs/app_reports/;
- gerar, opcionalmente, um relatório auxiliar com Gemini.

Para rodar:

bash streamlit run app/streamlit_app.py

Ou usando diretamente o Python do ambiente virtual:

bash ./.venv/bin/python -m streamlit run app/streamlit_app.py

No Windows:

bash .venv\Scripts\python -m streamlit run app/streamlit_app.py

A aplicação usa a mesma definição de success_commercial do pipeline principal e mantém os mesmos cuidados contra vazamento de dados.

## Gemini

A integração com Gemini é opcional.

O app funciona sem chave de API. Nesse caso, apenas a geração do relatório com LLM fica desativada.

Para habilitar:

bash cp .env.example .env

Depois edite o .env:

text GEMINI_API_KEY=sua_chave_aqui

O arquivo .env não deve ser versionado.

O Gemini só é chamado quando o usuário clica no botão de gerar relatório. O app envia apenas informações agregadas, como métricas, nomes dos modelos, melhor modelo, principais features e resumo do alvo. O dataset bruto não é enviado.

Esse relatório é um artefato auxiliar. O relatório final do projeto deve se basear principalmente nos outputs revisados, como:

text outputs/relatorio_base.md outputs/definicao_sucesso_comercial.md outputs/insights_acionaveis.md outputs/model_metrics.csv outputs/tables/statistical_tests.csv

## Modelos avaliados

O projeto compara modelos de diferentes níveis de complexidade:

- Regressão Logística;
- Árvore de Decisão;
- Random Forest;
- ExtraTreesClassifier;
- HistGradientBoostingClassifier.

XGBoost e LightGBM são opcionais. Se estiverem instalados, o pipeline pode testá-los; se não estiverem, eles são ignorados sem quebrar a execução.

## Melhor resultado atual

O melhor modelo atual é o hist_gradient_boosting.

Métricas no conjunto de teste:

text Accuracy:  0.8285 Precision: 0.6793 Recall:    0.8115 F1-score:  0.7395 ROC-AUC:   0.9155

As métricas completas ficam em:

text outputs/model_metrics.csv

## Como interpretar

O modelo identifica padrões associados ao sucesso comercial estimado em jogos já publicados na Steam.

Ele deve ser usado como apoio à decisão, não como garantia de sucesso. Os resultados ajudam a levantar hipóteses sobre produto, preço, posicionamento e recursos da Steam, mas precisam ser interpretados junto com contexto de gênero, público-alvo, orçamento e estratégia de lançamento.

Também é importante lembrar:

> associação não implica causalidade.

Ou seja, se jogos com multiplayer aparecem com maior taxa de sucesso estimado, isso não prova que multiplayer causa sucesso. Pode haver outros fatores envolvidos, como gênero, comunidade, marketing, orçamento ou maturidade do jogo.

## Principais outputs

text outputs/model_metrics.csv                 métricas finais dos modelos outputs/tables/model_cv_metrics.csv       validação cruzada outputs/tables/statistical_tests.csv      testes estatísticos outputs/tables/missing_values.csv         resumo de nulos outputs/tables/feature_importance.csv     importância de features outputs/figures/                          gráficos da EDA e dos modelos outputs/models/best_model.joblib          melhor modelo salvo localmente outputs/app_reports/                      saídas geradas pelo app Streamlit

## Limitações

A principal limitação é que a base não possui receita real por jogo. Por isso, success_commercial é uma proxy baseada em sinais públicos de alcance e engajamento.

Outros cuidados:

- algumas variáveis refletem contexto pós-lançamento;
- owners são estimativas em faixas;
- popularidade e exposição na Steam podem enviesar os resultados;
- tags e categorias podem refletir tanto posicionamento quanto popularidade;
- os resultados indicam associação, não causalidade.

Em resumo: o projeto não entrega uma fórmula de sucesso. Ele entrega um pipeline reprodutível para transformar dados públicos da Steam em evidências úteis para análise e decisão.