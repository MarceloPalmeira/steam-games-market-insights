# Definicao da variavel-alvo success_commercial

## Colunas usadas no score composto

A variavel-alvo `success_commercial` foi criada a partir de um score composto chamado `commercial_success_score`. As colunas usadas foram:

- `owners_midpoint`: ponto medio da faixa de `Estimated owners`;
- `total_reviews`: soma de avaliacoes positivas e negativas;
- `peak_ccu`: pico de jogadores simultaneos;
- `recommendations`: volume de recomendacoes registradas na Steam;
- `average_playtime_forever`: tempo medio de jogo acumulado.

## Por que Estimated owners nao foi usado sozinho

`Estimated owners` e uma proxy importante de alcance comercial, mas a coluna tem faixas largas e muitos empates. Na inspecao inicial, a faixa `0 - 20000` concentrava a maior parte da base, o que torna um corte direto por quartil pouco informativo. Usar essa coluna sozinha tambem ignoraria sinais relevantes de tracao, como reviews, engajamento e pico de usuarios.

## Como cada metrica entra no score

Cada componente passa por `log1p`, para reduzir o efeito de outliers extremos, e depois e convertido em ranking percentual dentro da base. O score final e a media simples desses rankings percentuais:

`commercial_success_score = media(rank_pct(log1p(metrica)))`

As metricas usadas no calculo foram: `owners_midpoint`, `total_reviews`, `peak_ccu`, `recommendations`, `average_playtime_forever`.

## Corte top 30%

O projeto usa uma formulacao de classificacao binaria. Um jogo recebe `success_commercial = 1` quando seu score composto esta no top 30% da base. O corte e calculado como o quantil 70 do `commercial_success_score`.

No ultimo calculo do pipeline, o corte foi `0.524124` e a proporcao positiva resultante foi 30.0%.

## Colunas removidas das features para evitar vazamento

As colunas diretamente usadas no alvo, ou derivadas diretamente delas, foram removidas da matriz de modelagem:

- `Estimated owners`
- `owners_lower`
- `owners_upper`
- `owners_midpoint`
- `Peak CCU`
- `peak_ccu`
- `Positive`
- `Negative`
- `positive_reviews`
- `negative_reviews`
- `total_reviews`
- `review_score_weighted`
- `positive_ratio`
- `Recommendations`
- `recommendations`
- `Average playtime forever`
- `average_playtime_forever`
- `commercial_success_score`
- `success_commercial`

`positive_ratio` tambem foi removida da matriz de modelagem por ser derivada diretamente de `Positive` e `Negative`. Ela permanece apenas na analise descritiva/EDA como variavel pos-lancamento para interpretar jogos ja publicados.

## Limitacoes da definicao

- O score e uma proxy de sucesso comercial, nao receita real observada.
- `Estimated owners` e informado em faixas, entao o ponto medio e uma aproximacao.
- Reviews e recomendacoes podem favorecer jogos antigos ou muito divulgados.
- `Peak CCU` e `Average playtime forever` medem engajamento, mas nao substituem vendas.
- O top 30% e uma escolha operacional para criar classes interpretaveis; outros cortes poderiam ser testados em analises futuras.

## Adequacao ao problema de negocio

Para um desenvolvedor indie ou publisher, sucesso comercial nao depende de um unico indicador. Alcance de mercado, volume de reviews, recomendacoes, pico de jogadores e engajamento capturam dimensoes complementares de tracao. A definicao composta ajuda a responder quais caracteristicas aparecem associadas a jogos que se destacam na Steam e evita depender exclusivamente de uma estimativa imperfeita de donos.
