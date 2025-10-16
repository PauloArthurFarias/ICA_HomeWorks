import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

def executar_pca_manual(dataframe, nome_dataset, pasta_destino):

    print(f"\nIniciando PCA para o dataset: {nome_dataset}...")

    X = dataframe.drop('quality', axis=1)
    y = dataframe['quality']

    # Pré-processamento: Padronizar os dados
    X_padronizado = StandardScaler().fit_transform(X)

    matriz_cov = np.cov(X_padronizado.T)

    autovalores, autovetores = np.linalg.eig(matriz_cov)

    # Selecionar os dois componentes principais
    indices_ordenados = np.argsort(autovalores)[::-1]
    componente_principal_1 = autovetores[:, indices_ordenados[0]]
    componente_principal_2 = autovetores[:, indices_ordenados[1]]

    # Projetar os dados nos componentes principais
    pc1 = X_padronizado.dot(componente_principal_1)
    pc2 = X_padronizado.dot(componente_principal_2)

    df_pca = pd.DataFrame(data={'PC1': pc1, 'PC2': pc2, 'quality': y})

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='quality', palette='viridis', alpha=0.7)
    plt.title(f'Análise de Componentes Principais (PCA) - {nome_dataset}')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    
    nome_arquivo = os.path.join(pasta_destino, f'{nome_dataset}_pca_plot.png')
    plt.savefig(nome_arquivo)
    plt.close()
    

# --- Execução Principal ---
if __name__ == "__main__":
    red_database = pd.read_csv('winequality-red.csv', sep=';')
    white_database = pd.read_csv('winequality-white.csv', sep=';')

    #TASK 01
    print(f'-RED- \n N:{red_database.shape[0]}, D:{red_database.shape[1]-1}, L:{red_database['quality'].nunique()} Class-Distribution: {red_database['quality'].value_counts()}\n')
    print(f'-WHITE- \n N:{white_database.shape[0]}, D:{white_database.shape[1]}, L:{white_database['quality'].nunique()} Class-Distribution: {white_database['quality'].value_counts()}\n')

    #TASK 02
    estatisticas_descritivas_red = red_database.describe()
    print("--- Resumo Estatístico de vinhos tintos (inclui Média e Desvio Padrão) ---")
    print(estatisticas_descritivas_red)

    estatisticas_descritivas_white = white_database.describe()
    print("--- Resumo Estatístico de vinhos brancos (inclui Média e Desvio Padrão) ---")
    print(estatisticas_descritivas_white)

    assimetria = white_database.skew(numeric_only=True)
    print("\n--- Assimetria (Skewness) de cada Variável ---")
    print(assimetria)

    # Seleciona apenas as colunas preditoras (exclui a coluna 'quality')
    variaveis_preditoras_red = red_database.drop('quality', axis=1)
    variaveis_preditoras_white = white_database.drop('quality', axis=1)

    pasta_graficos = 'graficos_task2'
    if not os.path.exists(pasta_graficos):
        print("Criando pasta para salvar os gráficos.")
        os.makedirs(pasta_graficos)

        for coluna in variaveis_preditoras_red.columns:
            plt.figure(figsize=(8, 5)) 
            sns.histplot(data=red_database, x=coluna, kde=True) # kde=True desenha uma linha suave de densidade
            plt.title(f'Histograma de "{coluna}"')
            plt.xlabel(coluna)
            plt.ylabel('Frequência')
            nome_arquivo = os.path.join(pasta_graficos, f'red_histogram_{coluna}.png')
            plt.savefig(nome_arquivo)
            plt.close()

        for coluna in variaveis_preditoras_red.columns:
            plt.figure(figsize=(6, 6))
            sns.boxplot(data=red_database, y=coluna)
            plt.title(f'Box-plot de "{coluna}"')
            plt.ylabel(coluna)
            nome_arquivo = os.path.join(pasta_graficos, f'red_boxplot_{coluna}.png')
            plt.savefig(nome_arquivo)
            plt.close()

        for coluna in variaveis_preditoras_white.columns:
            plt.figure(figsize=(8, 5)) 
            sns.histplot(data=white_database, x=coluna, kde=True)
            plt.title(f'Histograma de "{coluna}"')
            plt.xlabel(coluna)
            plt.ylabel('Frequência')
            nome_arquivo = os.path.join(pasta_graficos, f'white_histogram_{coluna}.png')
            plt.savefig(nome_arquivo)
            plt.close()

        for coluna in variaveis_preditoras_white.columns:
            plt.figure(figsize=(6, 6))
            sns.boxplot(data=white_database, y=coluna)
            plt.title(f'Box-plot de "{coluna}"')
            plt.ylabel(coluna)
            nome_arquivo = os.path.join(pasta_graficos, f'white_boxplot_{coluna}.png')
            plt.savefig(nome_arquivo)
            plt.close()
    else:
        print(f"\nA pasta '{pasta_graficos}' já existe. Os gráficos não serão recriados.")

    #TASK 03

    media_condicional_red = red_database.groupby('quality').mean()
    desvio_condicional_red = red_database.groupby('quality').std()
    assimetria_condicional_red = red_database.groupby('quality').skew()

    media_condicional_white = white_database.groupby('quality').mean()
    desvio_condicional_white = white_database.groupby('quality').std()
    assimetria_condicional_white = white_database.groupby('quality').skew()

    pasta_estatisticas = 'estatisticas_task3'
    if not os.path.exists(pasta_estatisticas):
        print("Criando pasta para salvar as tabelas de estatísticas.")
        os.makedirs(pasta_estatisticas)

        caminho_medias_red = os.path.join(pasta_estatisticas, 'tabela_medias_condicionais_red.csv')
        caminho_desvios_red = os.path.join(pasta_estatisticas, 'tabela_desvios_condicionais_red.csv')
        caminho_assimetrias_red = os.path.join(pasta_estatisticas, 'tabela_assimetrias_condicionais_red.csv')

        media_condicional_red.to_csv(caminho_medias_red)
        desvio_condicional_red.to_csv(caminho_desvios_red)
        assimetria_condicional_red.to_csv(caminho_assimetrias_red)

        caminho_medias_white = os.path.join(pasta_estatisticas, 'tabela_medias_condicionais_white.csv')
        caminho_desvios_white = os.path.join(pasta_estatisticas, 'tabela_desvios_condicionais_white.csv')
        caminho_assimetrias_white = os.path.join(pasta_estatisticas, 'tabela_assimetrias_condicionais_white.csv')

        media_condicional_white.to_csv(caminho_medias_white)
        desvio_condicional_white.to_csv(caminho_desvios_white)
        assimetria_condicional_white.to_csv(caminho_assimetrias_white)
    else:
        print(f"\nA pasta '{pasta_estatisticas}' já existe. Os gráficos não serão recriados.")

    estatisticas_condicionais_red = red_database.groupby('quality').agg(['mean', 'std', 'skew'])
    print("\n--- Tabela Completa de Estatísticas Condicionais Vinho Tinto ---")
    print(estatisticas_condicionais_red)

    estatisticas_condicionais_white = white_database.groupby('quality').agg(['mean', 'std', 'skew'])
    print("\n--- Tabela Completa de Estatísticas Condicionais Vinho Branco ---")
    print(estatisticas_condicionais_white)

    pasta_graficos = 'graficos_task3'
    if not os.path.exists(pasta_graficos):
        print("Criando pasta para salvar os gráficos.")
        os.makedirs(pasta_graficos)

        classes_red = sorted(red_database['quality'].unique())

        # Loop externo para as D variáveis preditoras
        for coluna in variaveis_preditoras_red.columns:
            # Loop interno para as L classes de qualidade
            for classe in classes_red:
                df_filtrado = red_database[red_database['quality'] == classe]
                
                plt.figure(figsize=(8, 5))
                sns.histplot(data=df_filtrado, x=coluna, kde=True)
                plt.title(f'Histograma de "{coluna}" para Qualidade {classe} (Tinto)')
                plt.xlabel(coluna)
                plt.ylabel('Frequência')
                nome_arquivo_hist = os.path.join(pasta_graficos, f'red_hist_{coluna}_quality_{classe}.png')
                plt.savefig(nome_arquivo_hist)
                plt.close()

                plt.figure(figsize=(6, 6))
                sns.boxplot(data=df_filtrado, y=coluna)
                plt.title(f'Box-plot de "{coluna}" para Qualidade {classe} (Tinto)')
                plt.ylabel(coluna)
                nome_arquivo_box = os.path.join(pasta_graficos, f'red_box_{coluna}_quality_{classe}.png')
                plt.savefig(nome_arquivo_box)
                plt.close()

        classes_white = sorted(white_database['quality'].unique())

        for coluna in variaveis_preditoras_white.columns:
            for classe in classes_white:
                df_filtrado = white_database[white_database['quality'] == classe]

                plt.figure(figsize=(8, 5))
                sns.histplot(data=df_filtrado, x=coluna, kde=True)
                plt.title(f'Histograma de "{coluna}" para Qualidade {classe} (Branco)')
                plt.xlabel(coluna)
                plt.ylabel('Frequência')
                nome_arquivo_hist = os.path.join(pasta_graficos, f'white_hist_{coluna}_quality_{classe}.png')
                plt.savefig(nome_arquivo_hist)
                plt.close()

                plt.figure(figsize=(6, 6))
                sns.boxplot(data=df_filtrado, y=coluna)
                plt.title(f'Box-plot de "{coluna}" para Qualidade {classe} (Branco)')
                plt.ylabel(coluna)
                nome_arquivo_box = os.path.join(pasta_graficos, f'white_box_{coluna}_quality_{classe}.png')
                plt.savefig(nome_arquivo_box)
                plt.close()

        print(f"\nProcesso finalizado. Todos os gráficos foram salvos no diretório '{pasta_graficos}'.")
    else:
        print(f"\nA pasta '{pasta_graficos}' já existe. Os gráficos não serão recriados.")

    #TASK 04

    pasta_graficos = 'graficos_task4'
    if not os.path.exists(pasta_graficos):
        print(f"Criando pasta para salvar os gráficos")
        os.makedirs(pasta_graficos)

        print("Gerando Heatmap de correlação para os Vinhos")
        matriz_correlacao_red = red_database.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(matriz_correlacao_red, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Matriz de Correlação para Vinho Tinto')
        caminho_heatmap_red = os.path.join(pasta_graficos, 'red_correlation_heatmap.png')
        plt.savefig(caminho_heatmap_red)
        plt.close()

        matriz_correlacao_white = white_database.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(matriz_correlacao_white, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Matriz de Correlação para Vinho Branco')
        caminho_heatmap_white = os.path.join(pasta_graficos, 'white_correlation_heatmap.png')
        plt.savefig(caminho_heatmap_white)
        plt.close()

        print("Gerando Pairplot para os Vinhos")
        pairplot_red = sns.pairplot(red_database, hue='quality', palette='viridis')
        caminho_pairplot_red = os.path.join(pasta_graficos, 'red_pairplot.png')
        pairplot_red.savefig(caminho_pairplot_red)
        plt.close()

        pairplot_white = sns.pairplot(white_database, hue='quality', palette='viridis')
        caminho_pairplot_white = os.path.join(pasta_graficos, 'white_pairplot.png')
        pairplot_white.savefig(caminho_pairplot_white)
        plt.close()

        print(f"\nProcesso finalizado. Todos os resultados foram salvos no diretório '{pasta_graficos}'.")
    else:
        print(f"\nA pasta '{pasta_graficos}' já existe. Os gráficos não serão recriados.")

    #TASK 05

    pasta_graficos = 'graficos_task5'
    if not os.path.exists(pasta_graficos):
        print(f"Criando o diretório para salvar os resultados: '{pasta_graficos}'")
        os.makedirs(pasta_graficos)
        executar_pca_manual(dataframe=red_database, nome_dataset="Vinho Tinto", pasta_destino=pasta_graficos)

        executar_pca_manual(dataframe=white_database, nome_dataset="Vinho Branco", pasta_destino=pasta_graficos)

        print("\nProcesso da Tarefa 5 finalizado.")
    else:
        print(f"\nA pasta '{pasta_graficos}' já existe. Os gráficos não serão recriados.")