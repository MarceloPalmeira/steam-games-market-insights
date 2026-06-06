# Steam Games Market Insights

Projeto final de Ciencia de Dados para predizer e explicar sucesso comercial de jogos na Steam, apoiando decisoes de desenvolvedores indie e publishers.

## Problema de negocio

Sou um desenvolvedor indie/publisher e quero entender como meu jogo pode performar no mercado da Steam e quais caracteristicas podem aumentar sua chance de sucesso comercial.

## Dataset

Fonte principal: `dataset/games.csv`, do Steam Games Dataset. O arquivo `dataset/games.json` fica disponivel como apoio, mas o pipeline usa o CSV como base principal.

Como `dataset/` esta no `.gitignore`, baixe a base manualmente no Kaggle e coloque os arquivos na pasta `dataset/` antes de executar o pipeline:

<https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?resource=download>

Estrutura esperada:

```text
dataset/
  games.csv
  games.json
```

Durante a leitura, o cabecalho `DiscountDLC count` e tratado como duas colunas (`Discount` e `DLC count`), porque as linhas possuem 40 campos. Essa correcao acontece apenas no carregamento; os arquivos brutos em `dataset/` nao sao editados.

## Variavel-alvo

`success_commercial` e uma proxy binaria de sucesso comercial, nao receita real. Ela identifica os jogos no top 30% de um score composto por:

- ponto medio de `Estimated owners`;
- total de reviews;
- pico de jogadores simultaneos;
- recomendacoes;
- tempo medio de jogo.

Cada metrica passa por `log1p`, ranking percentual e media simples. As colunas usadas diretamente no alvo, e derivadas diretas delas, sao removidas das features de modelagem para evitar vazamento. A explicacao completa fica em `outputs/definicao_sucesso_comercial.md`.

## Estrutura

- `src/`: codigo modular do pipeline.
- `dataset/`: dados brutos locais, nao versionados no GitHub.
- `outputs/figures/`: graficos gerados.
- `outputs/tables/`: tabelas, metricas, testes e importancias.
- `outputs/models/`: modelo salvo localmente.
- `outputs/relatorio_base.md`: base textual para o relatorio final.
- `outputs/insights_acionaveis.md`: recomendacoes praticas.
- `outputs/roteiro_apresentacao.md`: roteiro para pitch e apresentacao tecnica.

## Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Execucao

```bash
python3 -m src.main
```

## Modelos utilizados

- Regressao Logistica;
- Arvore de Decisao;
- Random Forest;
- ExtraTreesClassifier;
- HistGradientBoostingClassifier.

XGBoost e LightGBM sao opcionais: se estiverem instalados, o pipeline testa; se nao estiverem, sao pulados sem quebrar a execucao.

## Melhor modelo atual

O melhor modelo atual e `hist_gradient_boosting`, com aproximadamente:

- Accuracy: 0.8285;
- Precision: 0.6793;
- Recall: 0.8115;
- F1-score: 0.7395;
- ROC-AUC: 0.9155.

As metricas completas ficam em `outputs/model_metrics.csv`.

## Como interpretar os resultados

O modelo identifica padroes associados ao sucesso comercial historico na Steam. Os resultados devem apoiar decisoes de produto, preco, posicionamento e marketing, mas nao garantem sucesso futuro. Testes estatisticos e importancias indicam associacao, nao causalidade.

## Principais outputs

- `outputs/model_metrics.csv`: metricas finais dos modelos no teste.
- `outputs/tables/model_cv_metrics.csv`: validacao cruzada.
- `outputs/tables/statistical_tests.csv`: testes estatisticos.
- `outputs/tables/missing_values.csv`: resumo de nulos.
- `outputs/tables/feature_importance.csv`: permutation importance do melhor modelo quando necessario.
- `outputs/figures/`: graficos da EDA e explicabilidade.
- `outputs/models/best_model.joblib`: melhor modelo salvo localmente.

## Limitacoes

- A base nao possui receita real por jogo.
- `success_commercial` e uma proxy baseada em sinais de alcance e engajamento.
- Variaveis pos-lancamento ajudam a explicar jogos ja publicados, mas nao devem ser tratadas como garantidas antes do lancamento.
- Pode haver viés de popularidade e exposicao da plataforma.
- Correlação/associacao nao implica causalidade.

## Entrega e GitHub

O dataset e o modelo salvo sao grandes e podem ser regenerados localmente. Para GitHub, recomenda-se versionar codigo, README, relatorios e tabelas/figuras essenciais, mantendo `dataset/`, `.venv/`, caches e modelos binarios fora do versionamento.
