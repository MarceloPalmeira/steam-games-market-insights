Modelo Gemini usado: `gemini-2.5-flash`

# Relatório Executivo: Steam Games Market Insights

**Para:** Desenvolvedor Indie/Publisher
**De:** Consultoria de Ciência de Dados
**Data:** 18 de Maio de 2024
**Assunto:** Análise e Modelagem Preditiva para Sucesso Comercial de Jogos na Steam

---

## 1. Resumo do Problema de Negócio

O mercado de jogos na Steam é vasto e altamente competitivo, tornando desafiador para desenvolvedores e publishers independentes tomarem decisões eficazes de produto, preço, posicionamento e marketing. O objetivo principal deste projeto "Steam Games Market Insights" é fornecer insights preditivos baseados em dados para apoiar essas decisões estratégicas, visando otimizar as chances de sucesso comercial de novos lançamentos ou produtos existentes.

## 2. Definição da Variável-Alvo: `success_commercial`

Para este projeto, definimos uma variável-alvo chamada `success_commercial`. É crucial entender que esta é uma **proxy binária de sucesso comercial e não representa receita real**.

*   **Tipo:** Variável binária (0 ou 1), indicando se um jogo é considerado "comercialmente bem-sucedido" ou não.
*   **Construção:** Foi derivada a partir da agregação de métricas de desempenho in-game e de mercado, como `owners_midpoint` (estimativa de proprietários), `total_reviews` (total de avaliações), `peak_ccu` (pico de usuários simultâneos), `recommendations` (recomendações) e `average_playtime_forever` (tempo médio de jogo).
*   **Limiar:** O sucesso foi determinado aplicando um `cutoff` de 0.5241242629127891 em uma pontuação de sucesso comercial calculada, classificando aproximadamente os 30% (`top_share`: 0.3) dos jogos com melhor desempenho como "sucesso".
*   **Proporção de Sucesso:** Aproximadamente 30.00% (`positive_rate`: 0.30000570911255925) dos jogos no dataset são classificados como `success_commercial = 1`.

É importante salientar que nenhuma das componentes diretas desta proxy (e suas derivadas, como `owners_lower`, `owners_upper`, `positive_reviews`, `negative_reviews`, etc.) foram utilizadas como features de treino para evitar vazamento de dados (data leakage), garantindo que o modelo seja preditivo e não apenas descritivo.

## 3. Modelos Avaliados e Seleção do Melhor Modelo

Foram avaliados diversos modelos de Machine Learning para a tarefa de classificação binária de `success_commercial`. O dataset utilizado para treino e avaliação continha 122.611 linhas e 41 features. As métricas de desempenho obtidas na fase de validação foram as seguintes:

| Modelo                     | Accuracy | Precision | Recall | F1-Score | ROC AUC  |
| :------------------------- | :------- | :-------- | :----- | :------- | :------- |
| `hist_gradient_boosting`   | 0.8285   | 0.6793    | 0.8115 | 0.7395   | **0.9155** |
| `random_forest`            | 0.8307   | 0.6945    | 0.7780 | 0.7339   | 0.9117 |
| `extra_trees`              | 0.8143   | 0.6613    | 0.7809 | 0.7162   | 0.9008 |
| `decision_tree_baseline`   | 0.7949   | 0.6257    | 0.7877 | 0.6974   | 0.8880 |
| `logistic_regression_baseline` | 0.8008   | 0.6381    | 0.7763 | 0.7004   | 0.8862 |

Com base nas métricas, o modelo **`hist_gradient_boosting`** foi selecionado como o melhor modelo. Embora o `random_forest` tenha ligeiramente maior Accuracy e Precision, o `hist_gradient_boosting` demonstrou o **maior valor de ROC AUC (0.9155)**. O ROC AUC é uma métrica robusta que avalia a capacidade geral do modelo de distinguir entre as classes, sendo menos sensível ao desequilíbrio de classes e a limiares de classificação específicos. Isso o torna ideal para um problema onde identificar jogos potencialmente bem-sucedidos é crucial, minimizando tanto falsos positivos quanto falsos negativos ao longo de todos os limiares.

## 4. Interpretação das Métricas do Melhor Modelo (`hist_gradient_boosting`)

*   **Accuracy (Acurácia) = 0.8285:** O modelo acerta a classificação de sucesso ou não sucesso em aproximadamente 82.85% dos casos. Embora alta, esta métrica pode ser enganosa em problemas com classes desbalanceadas.
*   **Precision (Precisão) = 0.6793:** Dos jogos que o modelo previu como "comercialmente bem-sucedidos", cerca de 67.93% realmente o foram. Isso significa que aproximadamente 32% das previsões de sucesso foram falsos positivos (jogos previstos como sucesso, mas que não atingiram o limiar). Para o desenvolvedor, isso significa que nem todo jogo apontado como promissor será de fato um sucesso, e investimentos baseados nessa previsão podem ter um risco de 32% de serem mal direcionados.
*   **Recall (Sensibilidade) = 0.8115:** Dos jogos que *realmente* foram "comercialmente bem-sucedidos", o modelo identificou corretamente 81.15%. Isso significa que cerca de 19% dos jogos de sucesso foram "perdidos" pelo modelo (falsos negativos). Para o desenvolvedor, isso representa uma taxa de "oportunidades perdidas" de aproximadamente 19%, onde jogos com potencial de sucesso não foram identificados pelo modelo.
*   **F1-Score = 0.7395:** O F1-Score é uma média harmônica entre Precision e Recall, oferecendo um bom equilíbrio entre as duas métricas. Um valor de quase 0.74 indica um bom balanceamento entre a capacidade do modelo de identificar sucessos verdadeiros e de evitar falsos positivos.
*   **ROC AUC = 0.9155:** Este é um resultado excelente. Um ROC AUC acima de 0.9 indica que o modelo tem uma capacidade discriminatória muito forte, sendo capaz de distinguir entre jogos de sucesso e não-sucesso de forma consistente em diferentes limiares de probabilidade.

Em resumo, o modelo `hist_gradient_boosting` apresenta uma performance robusta, com alta capacidade de identificação de jogos de sucesso (Recall) e uma boa taxa de acertos nas suas previsões (Precision), suportada por uma excelente capacidade discriminatória geral (ROC AUC).

## 5. Principais Features (Características)

As informações agregadas fornecidas **não contêm a lista das principais features (características) ou suas importâncias** para o modelo. Esta é uma limitação significativa para fornecer recomendações muito específicas.

No entanto, podemos inferir que o modelo utiliza as 41 features fornecidas no dataset de treino (com exclusão das colunas de vazamento de dados como `owners_midpoint`, `total_reviews`, `peak_ccu`, `recommendations`, `average_playtime_forever`, `positive_ratio`, `Positive`, `Negative`, etc.).

Para futuras análises, seria fundamental extrair e analisar as importâncias das features para entender quais aspectos (ex: gênero, tags, preço inicial, presença de multiplayer, data de lançamento, experiência da desenvolvedora, etc.) contribuem mais para a previsão de sucesso comercial.

## 6. Recomendações Acionáveis

Mesmo sem a granularidade das features, o modelo oferece insights valiosos:

1.  **Utilização Preditiva em Fases Iniciais:** Implemente o modelo `hist_gradient_boosting` para prever a probabilidade de `success_commercial` para **protótipos de jogos, conceitos iniciais ou jogos em desenvolvimento**. Isso pode ajudar a validar ideias, identificar potenciais "red flags" (sinais de alerta) ou "green flags" (sinais promissores) e orientar ajustes no design do jogo, modelo de monetização ou estratégia de lançamento antes de grandes investimentos.
2.  **Análise Comparativa e Iterativa:** Utilize o modelo para avaliar diferentes cenários de produto (e.g., "o que aconteceria se adicionássemos multiplayer?", "e se o preço fosse X versus Y?"). Embora as features que representam esses cenários não estejam explicitamente detalhadas no relatório, a capacidade do modelo de prever o sucesso pode ser usada para comparar a probabilidade de sucesso entre diferentes configurações ou estratégias hipotéticas, exigindo mais análise sobre como essas estratégias se traduzem nas features do modelo.
3.  **Priorização de Investimentos em Marketing:** O `precision` de 67.93% indica que o modelo tem uma boa, mas não perfeita, capacidade de identificar sucessos. Use as previsões do modelo para **priorizar esforços de marketing e alocação de recursos**. Para jogos com alta probabilidade de sucesso segundo o modelo, pode-se justificar um investimento maior. Para jogos com menor probabilidade, pode-se optar por uma abordagem mais conservadora ou reavaliar o projeto, sempre combinando a informação quantitativa com o julgamento humano e pesquisa de mercado qualitativa.

## 7. Limitações e Cuidados na Interpretação

*   **Proxy de Sucesso:** A variável `success_commercial` é uma **proxy**. Ela não mede receita real ou lucro, apenas um conjunto de indicadores de desempenho que tendem a estar associados a sucesso. Um jogo pode ter alta pontuação na proxy mas não ser lucrativo, ou vice-versa, dependendo de fatores externos não capturados.
*   **Não Implica Causalidade:** Os modelos de Machine Learning identificam associações e padrões nos dados. As previsões do modelo indicam a *probabilidade* de um resultado com base em características observadas, mas **não estabelecem relações de causalidade**. Por exemplo, se o modelo predizer que jogos com "Tag X" são mais bem-sucedidos, isso não significa que adicionar "Tag X" *causará* o sucesso, mas sim que essa tag está *associada* a jogos que tiveram sucesso no passado.
*   **Ausência de Features Importantes:** A maior limitação deste relatório é a **falta de detalhamento das features mais importantes**. Sem essa informação, as recomendações não podem ser específicas sobre quais aspectos do jogo (gênero, arte, mecânicas, preço, marketing pré-lançamento etc.) o modelo considera mais impactantes. Uma análise de feature importance é crucial para a próxima etapa.
*   **Vazamento de Dados (Leakage):** As métricas de sucesso direto (proprietários, reviews, CCU, etc.) foram removidas das features de treino para garantir que o modelo seja preditivo. Isso significa que o modelo não pode usar essas informações diretas para fazer uma previsão.
*   **Mercado Dinâmico:** O mercado de jogos da Steam está em constante evolução. Tendências de gêneros, expectativas dos jogadores, estratégias de marketing e modelos de precificação mudam ao longo do tempo. O modelo foi treinado com dados históricos e sua performance pode degradar se as dinâmicas do mercado mudarem significativamente. Recomenda-se retreinamento periódico com dados atualizados.
*   **Generalização:** O modelo foi treinado em mais de 120.000 jogos. Embora robusto, pode haver nuances específicas para um nicho muito particular de jogos indie que podem não ser totalmente capturadas. É sempre recomendável combinar a previsão do modelo com expertise humana e conhecimento específico do segmento de mercado.