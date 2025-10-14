import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

pasta_graficos = 'graficos'
if not os.path.exists(pasta_graficos):
    print("Criando pasta 'graficos' para salvar os gráficos.")
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
    print("A pasta 'graficos' já existe. Os gráficos não serão recriados.")