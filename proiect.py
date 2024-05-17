import numpy as np
import pandas as pd

train_df = pd.read_csv('train.csv')
nr_rows = len(train_df.axes[0])
nr_columns = len(train_df.axes[1])

print("Numarul de linii:", nr_rows)
print("Numarul de coloane:", nr_columns, "\n")
print("Tipul de date al fiecarei coloane:\n", train_df.dtypes, "\n")
print("Numarul de valori lipsa pentru fiecare coloana: \n", train_df.isnull().sum(), "\n")
print("Numarul de linii duplicate: ", train_df.duplicated().sum())
