# ICA_HomeWorks
Repositório pra cooperação no processo de conclusão do HomeWork da disciplina de Inteligência Computacional Aplicada. Equipe: Paulo Arthur, Gabriel Damasceno, Lucca Dall'olio.

# Análise Exploratória e PCA de Qualidade de Vinhos

Este projeto realiza uma Análise Exploratória de Dados (EDA) e uma Análise de Componentes Principais (PCA) manual em dois datasets de qualidade de vinhos: tinto (`winequality-red.csv`) e branco (`winequality-white.csv`).

O script é dividido em várias tarefas (TASKS) que executam o seguinte:
* **Task 01:** Imprime estatísticas básicas dos datasets (número de amostras, dimensões, distribuição de classes).
* **Task 02:** Calcula estatísticas descritivas (média, desvio padrão, assimetria) e gera visualizações (histogramas, boxplots) para todas as variáveis preditoras. Salva os gráficos na pasta `graficos_task2/`.
* **Task 03:** Calcula estatísticas descritivas condicionais (agrupadas pela variável 'quality') e salva os resultados em tabelas CSV na pasta `estatisticas_task3/`. Também gera histogramas e boxplots condicionais para cada variável e classe, salvando-os em `graficos_task3/`.
* **Task 04:** Gera e salva heatmaps de correlação e pairplots (coloridos por qualidade) para ambos os datasets na pasta `graficos_task4/`.
* **Task 05:** Executa uma implementação manual de PCA, reduzindo os dados para 2 componentes principais. Salva os gráficos resultantes (scatter plot dos componentes e biplot dos vetores) na pasta `graficos_task5/`.

## Bibliotecas Necessárias

O projeto utiliza as seguintes bibliotecas Python:

* **pandas:** Para manipulação e análise de dados (leitura de CSV, dataframes).
* **numpy:** Para operações numéricas e de álgebra linear (cálculo de matriz de covariância, autovalores/autovetores).
* **matplotlib:** Para a geração de gráficos base.
* **seaborn:** Para visualização estatística de dados (histogramas, boxplots, heatmaps, pairplots).
* **scikit-learn:** Especificamente `StandardScaler` para a padronização dos dados antes do PCA.

## Como Rodar

1.  **Pré-requisitos:**
    * Tenha o Python 3.x instalado.
    * Tenha os arquivos de dados `winequality-red.csv` e `winequality-white.csv` no mesmo diretório que o script.

2.  **Instalação das Dependências:**
    Você pode instalar todas as bibliotecas necessárias usando o `pip`:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Execução:**
    Salve o código fornecido em um arquivo (por exemplo, `analise_vinhos.py`) e execute-o através do terminal:
    ```bash
    python analise_vinhos.py
    ```

## Saída (Output)

Ao ser executado, o script fará o seguinte:

1.  Imprimirá diversas estatísticas descritivas e resumos diretamente no console.
2.  Criará (se não existirem) as seguintes pastas no diretório atual:
    * `graficos_task2/`
    * `estatisticas_task3/`
    * `graficos_task3/`
    * `graficos_task4/`
    * `graficos_task5/`
3.  Povoará essas pastas com os arquivos `.png` (gráficos) e `.csv` (tabelas de estatísticas) correspondentes a cada tarefa.

**Nota:** O script verifica se as pastas de saída já existem. Se uma pasta for encontrada, a geração de arquivos para essa tarefa específica será pulada, e uma mensagem será impressa no console (ex: "A pasta 'graficos_task2' já existe. Os gráficos não serão recriados.").