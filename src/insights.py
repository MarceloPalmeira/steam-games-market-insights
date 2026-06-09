from __future__ import annotations

import pandas as pd

from src.config import OUTPUT_DIR, SCORE_COLUMN, TARGET_COLUMN


def _pct(value: float | int) -> str:
    if pd.isna(value):
        return "n/d"
    return f"{100 * float(value):.1f}%"


def _p_value(value: float | int) -> str:
    if pd.isna(value):
        return "n/d"
    number = float(value)
    if number == 0:
        return "<1e-300"
    if number < 0.0001:
        return f"{number:.2e}"
    return f"{number:.4f}"


def _group_rate(group_comparisons: pd.DataFrame, group: str, value: int, column: str = "success_rate") -> float:
    row = group_comparisons.loc[
        (group_comparisons["group"] == group) & (group_comparisons["value"] == value)
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def write_success_definition(target_info: dict[str, object]) -> None:
    components = target_info["components"]
    leakage_columns = target_info["leakage_columns_removed"]
    cutoff = float(target_info["cutoff"])
    positive_rate = float(target_info["positive_rate"])

    content = f"""# Definicao da variavel-alvo success_commercial

## Colunas usadas no score composto

A variavel-alvo `success_commercial` foi criada a partir de um score composto chamado `{SCORE_COLUMN}`. As colunas usadas foram:

- `owners_midpoint`: ponto medio da faixa de `Estimated owners`;
- `total_reviews`: soma de avaliacoes positivas e negativas;
- `peak_ccu`: pico de jogadores simultaneos;
- `recommendations`: volume de recomendacoes registradas na Steam;
- `average_playtime_forever`: tempo medio de jogo acumulado.

## Por que Estimated owners nao foi usado sozinho

`Estimated owners` e uma proxy importante de alcance comercial, mas a coluna tem faixas largas e muitos empates. Na inspecao inicial, a faixa `0 - 20000` concentrava a maior parte da base, o que torna um corte direto por quartil pouco informativo. Usar essa coluna sozinha tambem ignoraria sinais relevantes de tracao, como reviews, engajamento e pico de usuarios.

## Como cada metrica entra no score

Cada componente passa por `log1p`, para reduzir o efeito de outliers extremos, e depois e convertido em ranking percentual dentro da base. O score final e a media simples desses rankings percentuais:

`{SCORE_COLUMN} = media(rank_pct(log1p(metrica)))`

As metricas usadas no calculo foram: {", ".join(f"`{component}`" for component in components)}.

## Corte top 30%

O projeto usa uma formulacao de classificacao binaria. Um jogo recebe `success_commercial = 1` quando seu score composto esta no top 30% da base. O corte e calculado como o quantil 70 do `{SCORE_COLUMN}`.

No ultimo calculo do pipeline, o corte foi `{cutoff:.6f}` e a proporcao positiva resultante foi {_pct(positive_rate)}.

## Colunas removidas das features para evitar vazamento

As colunas diretamente usadas no alvo, ou derivadas diretamente delas, foram removidas da matriz de modelagem:

{chr(10).join(f"- `{column}`" for column in leakage_columns)}

`positive_ratio` tambem foi removida da matriz de modelagem por ser derivada diretamente de `Positive` e `Negative`. Ela permanece apenas na analise descritiva/EDA como variavel pos-lancamento para interpretar jogos ja publicados.

## Limitacoes da definicao

- O score e uma proxy de sucesso comercial, nao receita real observada.
- `Estimated owners` e informado em faixas, entao o ponto medio e uma aproximacao.
- Reviews e recomendacoes podem favorecer jogos antigos ou muito divulgados.
- `Peak CCU` e `Average playtime forever` medem engajamento, mas nao substituem vendas.
- O top 30% e uma escolha operacional para criar classes interpretaveis; outros cortes poderiam ser testados em analises futuras.

## Adequacao ao problema de negocio

Para um desenvolvedor indie ou publisher, sucesso comercial nao depende de um unico indicador. Alcance de mercado, volume de reviews, recomendacoes, pico de jogadores e engajamento capturam dimensoes complementares de tracao. A definicao composta ajuda a responder quais caracteristicas aparecem associadas a jogos que se destacam na Steam e evita depender exclusivamente de uma estimativa imperfeita de donos.
"""
    (OUTPUT_DIR / "definicao_sucesso_comercial.md").write_text(content, encoding="utf-8")


def build_actionable_insights(
    group_comparisons: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> list[str]:
    multiplayer_yes = _group_rate(group_comparisons, "Multiplayer", 1)
    multiplayer_no = _group_rate(group_comparisons, "Multiplayer", 0)
    free_yes = _group_rate(group_comparisons, "Gratuito", 1)
    free_no = _group_rate(group_comparisons, "Gratuito", 0)
    achievements_yes = _group_rate(group_comparisons, "Com achievements", 1)
    achievements_no = _group_rate(group_comparisons, "Com achievements", 0)

    top_features = ", ".join(feature_importance.head(5)["feature"].tolist())

    return [
        (
            f"Percebemos que jogos com multiplayer apresentaram taxa de sucesso de {_pct(multiplayer_yes)}, "
            f"contra {_pct(multiplayer_no)} nos demais jogos. Isso sugere que recursos sociais podem ampliar "
            "retencao, descoberta e recorrencia de uso. Por isso, recomendamos avaliar modos cooperativos, "
            "competitivos ou eventos online quando eles fizerem sentido para o genero. A decisao foi sustentada "
            "pela comparacao de taxa de sucesso e pelo teste estatistico registrado em `outputs/tables/statistical_tests.csv`. "
            "Apos 6 meses do lancamento, o sucesso pode ser avaliado por pico de usuarios simultaneos, reviews e retencao. "
            "A acao sera considerada bem-sucedida se o jogo superar em pelo menos 10% a taxa media de sucesso do seu segmento."
        ),
        (
            f"Percebemos que jogos gratuitos tiveram taxa de sucesso de {_pct(free_yes)}, enquanto jogos pagos tiveram "
            f"{_pct(free_no)}. Isso sugere que preco zero pode aumentar alcance, mas precisa ser analisado junto ao modelo "
            "de monetizacao e ao genero. Por isso, recomendamos testar precos de entrada baixos, demos ou periodos promocionais "
            "sem depender automaticamente de free-to-play. A decisao foi sustentada pela comparacao entre grupos gratuito/pago. "
            "Apos 3 meses, o sucesso pode ser avaliado por conversao, reviews, donos estimados e receita por usuario. "
            "A acao sera considerada bem-sucedida se aumentar alcance sem reduzir a receita estimada por usuario abaixo da meta do projeto."
        ),
        (
            f"Percebemos que jogos com achievements tiveram taxa de sucesso de {_pct(achievements_yes)}, contra "
            f"{_pct(achievements_no)} nos demais. Isso sugere que sistemas de progresso podem reforcar engajamento e "
            "sinalizar acabamento do produto. Por isso, recomendamos planejar achievements conectados a marcos reais de gameplay, "
            "evitando conquistas artificiais. A decisao foi sustentada pela comparacao entre grupos e pelas importancias do modelo, "
            f"cujas variaveis mais relevantes incluem: {top_features}. Apos 3 meses, o sucesso pode ser avaliado por tempo medio de jogo "
            "e proporcao de jogadores que desbloqueiam conquistas intermediarias. A acao sera considerada bem-sucedida se elevar o tempo medio "
            "de jogo e a taxa de reviews positivas sem aumentar reclamacoes sobre repetitividade."
        ),
    ]


def write_insights(insights: list[str]) -> None:
    content = "# Insights acionaveis\n\n" + "\n\n".join(
        f"## Insight {index}\n\n{insight}" for index, insight in enumerate(insights, start=1)
    )
    (OUTPUT_DIR / "insights_acionaveis.md").write_text(content, encoding="utf-8")


def write_report_base(
    target_info: dict[str, object],
    metrics_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    insights: list[str],
    dataset_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    best = metrics_df.iloc[0].to_dict()
    metrics_text = _markdown_table(metrics_df)
    cv_text = _markdown_table(cv_df)
    rows = _summary_value(dataset_summary, "rows")
    engineered_columns = _summary_value(dataset_summary, "columns_after_engineering")
    top_features = ", ".join(feature_importance.head(5)["feature"].tolist())
    tests_text = "\n".join(
        f"- {row['test']}: {row['description']} (p-valor={_p_value(row['p_value'])})"
        for _, row in statistical_tests.iterrows()
    )
    insights_text = "\n".join(f"- {insight}" for insight in insights)

    content = f"""# Relatorio-base - Predicao de sucesso comercial de jogos Steam

## Aplicacao

O problema de negocio e apoiar um desenvolvedor indie ou publisher a entender quais caracteristicas estao associadas a melhor desempenho comercial na Steam. O projeto usa dados historicos de jogos para predizer e explicar `success_commercial`.

## Base de Dados

A base principal e `dataset/games.csv`, do Steam Games Dataset. O pipeline processou {int(rows):,} registros e {int(engineered_columns)} colunas apos a engenharia de atributos. Durante a leitura, o cabecalho `DiscountDLC count` e tratado como duas colunas: `Discount` e `DLC count`, porque as linhas possuem 40 campos. O CSV bruto nao e editado. O feedback do monitor foi incorporado pela definicao explicita da variavel-alvo.

As principais variaveis originais usadas na analise incluem preco, data de lancamento, plataformas, estimated owners, reviews positivas/negativas, peak CCU, achievements, metacritic, categorias, generos e tags. A preparacao inclui conversao numerica, parsing de datas, parsing de faixas de owners, tratamento de nulos por imputacao nos modelos, transformacao logaritmica de variaveis assimetricas e criacao de indicadores binarios para plataformas e caracteristicas de gameplay.

## Estatistica Descritiva e Inferencia

Foram geradas distribuicoes de preco, reviews, score comercial, grupos multiplayer/gratuito/achievements e matriz de correlacao. As visualizacoes respondem perguntas de negocio: tamanho do grupo de sucesso, diferenca entre jogos pagos e gratuitos, associacao de multiplayer com sucesso, concentracao de reviews e relacoes entre variaveis numericas. Os testes estatisticos principais foram:

{tests_text}

## Metodos Avaliados

Foram avaliados uma Regressao Logistica como baseline, uma Arvore de Decisao simples, Random Forest, Extra Trees e HistGradientBoosting como modelos avancados. XGBoost e LightGBM sao treinados apenas se estiverem instalados no ambiente.

## Metricas de Avaliacao

As metricas usadas foram accuracy, precision, recall, F1-score, ROC-AUC e matriz de confusao. O modelo vencedor foi `{best['model']}`.

## Metodos de Avaliacao

A avaliacao usa divisao treino/teste estratificada, com teste de {20}% da base, e validacao cruzada estratificada com 3 folds no conjunto de treino.

## Resultados e Discussao

Metricas no conjunto de teste:

{metrics_text}

Metricas de validacao cruzada:

{cv_text}

A variavel-alvo foi definida como top 30% do score composto. O corte calculado foi `{float(target_info['cutoff']):.6f}` e a taxa positiva foi {_pct(float(target_info['positive_rate']))}.

## Explicabilidade

Como o melhor modelo atual e baseado em HistGradientBoosting, a explicabilidade usa permutation importance. As variaveis mais relevantes no pipeline final foram: {top_features}. Em linguagem de negocio, isso indica que variedade/quantidade de tags, idade do jogo, sinais de acabamento/recursos da Steam, preco e aprovacao observada em EDA ajudam a explicar diferencas de desempenho na base.

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

{insights_text}

## Bibliografia/Referencias

- Steam Games Dataset, Kaggle.
- Especificacao do Projeto Final de Ciencia de Dados, UFAL, 2026.
- Scikit-learn documentation.
- SciPy documentation.
"""
    (OUTPUT_DIR / "relatorio_base.md").write_text(content, encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Tabela vazia._"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + rows)


def _summary_value(summary_df: pd.DataFrame, metric: str) -> float:
    row = summary_df.loc[summary_df["metric"] == metric, "value"]
    if row.empty:
        return float("nan")
    return float(row.iloc[0])


def write_presentation_script(
    metrics_df: pd.DataFrame,
    insights: list[str],
    feature_importance: pd.DataFrame,
) -> None:
    best = metrics_df.iloc[0]
    top_features = ", ".join(feature_importance.head(5)["feature"].tolist())
    insights_text = "\n".join(f"- {item}" for item in insights)
    content = f"""# Roteiro de apresentacao

## Formato oficial

- 5 minutos para o pitch, com foco comercial no problema, solução e impacto.
- 5 minutos para perguntas técnicas sobre o projeto.
- O conteúdo técnico abaixo deve ser usado como preparação para perguntas, não como bloco cronometrado separado.

## Pitch - 5 minutos

### Slide 1 - Problema e público-alvo
Fala sugerida: Desenvolvedores indie e publishers precisam decidir quais características priorizar a partir de evidências associadas ao destaque de jogos na Steam, um mercado competitivo e com forte assimetria de popularidade.

### Slide 2 - Base de dados e oportunidade
Fala sugerida: A base Steam Games Dataset, disponível no Kaggle, permite combinar preço, data de lançamento, plataformas, owners estimados, reviews, peak CCU, categorias, gêneros, tags e achievements para transformar histórico de mercado em apoio à decisão.

### Slide 3 - Variável-alvo `success_commercial`
Fala sugerida: Como a base não possui receita real, definimos `success_commercial` como uma proxy: jogos no top 30% de um score composto por owners estimados, total de reviews, peak CCU, recomendações e tempo médio de jogo. As colunas usadas no alvo foram removidas das features para evitar vazamento.

### Slide 4 - Evidências principais da EDA
Fala sugerida: As comparações entre grupos indicam associações relevantes: jogos com multiplayer tiveram taxa de sucesso de 44.9%, contra 26.7% nos demais; jogos pagos tiveram taxa maior que gratuitos; jogos com achievements também tiveram taxa maior que jogos sem achievements.

### Slide 5 - Modelos e melhor resultado
Fala sugerida: Avaliamos Regressão Logística, Árvore de Decisão, Random Forest, Extra Trees e HistGradientBoosting. O melhor modelo atual foi `{best['model']}`, com Accuracy={best['accuracy']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, F1={best['f1']:.4f} e ROC-AUC={best['roc_auc']:.4f}.

### Slide 6 - Insights acionáveis
Fala sugerida: Recomendamos avaliar recursos sociais quando fizerem sentido ao gênero, testar estratégia de preço sem assumir automaticamente free-to-play e planejar achievements conectados ao progresso real do jogo. Essas recomendações devem ser tratadas como hipóteses de decisão apoiadas por evidências, não como garantias de sucesso.

### Slide 7 - App Streamlit/Gemini como recurso extra
Fala sugerida: O projeto também possui uma aplicação local em Streamlit para carregar o dataset, selecionar modelos, visualizar métricas/gráficos e gerar um relatório automatizado com Gemini. O Gemini é opcional via `GEMINI_API_KEY`, e o app funciona mesmo sem chave de API.

### Slide 8 - Limitações e decisão recomendada
Fala sugerida: A base não possui receita real, o alvo é uma proxy, algumas variáveis são pós-lançamento e as evidências indicam associação, não causalidade. A decisão recomendada é usar o modelo como apoio para priorizar produto, preço e posicionamento, sempre combinando os resultados com análise de gênero, público e estratégia de lançamento.

## Preparação para perguntas técnicas

- Base: Steam Games Dataset do Kaggle, com `dataset/games.csv` como fonte principal e `dataset/` fora do versionamento.
- Preparação: correção do cabeçalho `DiscountDLC count` no carregamento, conversão numérica, parsing de datas, parsing de owners, parsing de listas/tags/gêneros/categorias e criação de features.
- Alvo: `success_commercial` é proxy top 30% do score composto; não é receita real.
- Vazamento: `Estimated owners`, owners derivados, `Positive`, `Negative`, `total_reviews`, `positive_ratio`, `Peak CCU`, `Recommendations`, `Average playtime forever`, score composto e alvo foram removidos das features.
- Inferência: Mann-Whitney U e qui-quadrado com p-valores; interpretação sempre como associação estatística.
- Validação: split treino/teste estratificado com 20% para teste e validação cruzada estratificada com 3 folds no treino.
- Erros: falsos positivos podem levar a expectativa excessiva de sucesso; falsos negativos podem esconder oportunidades promissoras. O modelo apoia decisão, não substitui avaliação humana.
- App extra: Streamlit com seleção de modelos, gráficos Matplotlib, relatório Gemini opcional e outputs em `outputs/app_reports/`.

## Possíveis perguntas da banca

- Por que não usar apenas Estimated owners? Porque a coluna tem faixas largas e muitos empates; o score composto usa sinais complementares de alcance, reviews e engajamento.
- Há vazamento de dados? As colunas usadas diretamente no alvo e derivadas diretas foram removidas das features; `positive_ratio` permanece apenas na EDA.
- O modelo serve para jogos antes do lançamento? Parcialmente. Para uso pré-lançamento, seria necessário restringir as features a informações disponíveis antes do lançamento.
- Por que top 30%? É um corte operacional que cria uma classe positiva relevante e ainda com tamanho suficiente para treino e avaliação.
- Correlação implica causalidade? Não. Os resultados indicam associações e ajudam a priorizar hipóteses de decisão e experimentação.
- O que o app acrescenta? Ele facilita exploração, comparação de modelos e geração de relatório automatizado, funcionando como recurso extra da entrega.
"""
    (OUTPUT_DIR / "roteiro_apresentacao.md").write_text(content, encoding="utf-8")
