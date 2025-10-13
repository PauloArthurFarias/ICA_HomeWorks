import pandas as pd
import numpy as np

red_database = pd.read_csv('winequality-red.csv', sep=';')
white_database = pd.read_csv('winequality-white.csv', sep=';')

#TASK 01
print(f'-RED- \n N:{red_database.shape[0]}, D:{red_database.shape[1]}, L:{red_database['quality'].nunique()} Class-Distribution: {red_database['quality'].value_counts()}\n')
print(f'-WHITE- \n N:{white_database.shape[0]}, D:{white_database.shape[1]}, L:{white_database['quality'].nunique()} Class-Distribution: {white_database['quality'].value_counts()}\n')

