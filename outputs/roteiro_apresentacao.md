# Roteiro de apresentacao

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
Fala sugerida: Avaliamos Regressão Logística, Árvore de Decisão, Random Forest, Extra Trees e HistGradientBoosting. O melhor modelo atual foi `hist_gradient_boosting`, com Accuracy=0.8285, Precision=0.6793, Recall=0.8115, F1=0.7395 e ROC-AUC=0.9155.

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
