import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_1():
    print("-------------- TASK 1 --------------\n")

    print(f"Numarul de coloane: {nr_columns}\n")
    print(f"Tipul de date al fiecarei coloane:\n{train_df.dtypes}\n")
    print(f"Numarul de valori lipsa pentru fiecare coloana:\n{train_df.isnull().sum()}\n")

    print(f"Numarul de linii: {nr_rows}")
    print(f"Numarul de linii duplicate: {train_df.duplicated().sum()}\n")

def task_2():
    print("\n-------------- TASK 2 --------------\n")

    # Procentul de supravietuitori
    nr_survived = (train_df['Survived']).sum()
    survived_percentage = (nr_survived / nr_rows)
    print(f"Procentul persoanelor care au supravietuit: {survived_percentage : %}")
    print(f"Procentul persoanelor care nu au supravietuit: {1 - survived_percentage : %}\n")

    # Procentul pasagerilor din fiecare clasa
    classes = train_df['Pclass']
    unique_classes = classes.unique()
    unique_classes.sort()
    class_count = np.zeros(len(unique_classes) + 1, dtype=int)

    for x in classes:
        class_count[x] += 1

    class_percentage = class_count[1:] / nr_rows

    for x in unique_classes:
        print(f"Clasa {x}: {class_percentage[x - 1] : %}")

    # Procentul de barbati si de femei
    genders = train_df['Sex']
    unique_genders = genders.unique()
    unique_genders.sort()
    genders_count = np.zeros(len(unique_genders), dtype=int)

    for x in genders:
        if x == 'female':
            genders_count[0] += 1
        else:
            genders_count[1] += 1

    genders_percentage = genders_count / nr_rows

    print(f"\nProcentul de barbati: {genders_percentage[1] : %}")
    print(f"Procentul de femei: {genders_percentage[0] : %}\n")

    # Realizam graficele pentru reprezentarea rezultatelor
    survived_array = [1 - survived_percentage, survived_percentage]
    plt.figure()
    plt.bar(["False", "True"], survived_array)
    plt.xticks(["False", "True"])
    plt.ylabel("Percentage")
    plt.xlabel("Survived")
    plt.savefig("task2-plots/Survivors.png")
    plt.show()

    plt.figure()
    plt.bar(unique_classes, class_percentage)
    plt.xticks(unique_classes)
    plt.ylabel("Percentage")
    plt.xlabel("Classes")
    plt.savefig("task2-plots/Classes.png")
    plt.show()

    plt.figure()
    plt.bar(unique_genders, genders_percentage)
    plt.ylabel("Percentage")
    plt.xlabel("Gender")
    plt.savefig('task2-plots/Male-Female.png')
    plt.show()

# Determinam ce task vrem sa rulam
# Daca nu se specifica un task, se vor executa toate
task = 0
if len(sys.argv) > 1:
    task = int(sys.argv[1])

train_df = pd.read_csv('input/train.csv')
nr_columns = len(train_df.axes[1])
nr_rows = len(train_df.axes[0])

# Task 1
if task == 1 or task == 0:
    task_1()

# Task 2
if task == 2 or task == 0:
    task_2()
