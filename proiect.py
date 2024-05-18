import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_1():
    print("-------------- TASK 1 --------------\n")

    print(f"Number of columns: {nr_columns}\n")
    print(f"Columns data types:\n{train_df.dtypes}\n")
    print(f"Number of missing values for each column:\n{train_df.isnull().sum()}\n")

    print(f"Number of rows: {nr_rows}")
    print(f"Number of duplicate rows: {train_df.duplicated().sum()}\n")

def task_2():
    print("\n-------------- TASK 2 --------------\n")

    # Procentul de supravietuitori
    nr_survived = (train_df['Survived']).sum()
    survived_percentage = (nr_survived / nr_rows)
    print(f"Percentage of people that survived: {survived_percentage : %}")
    print(f"Percentage of people that didn't survive: {1 - survived_percentage : %}\n")

    # Procentul pasagerilor din fiecare clasa
    classes = train_df['Pclass']
    unique_classes = classes.unique()
    unique_classes.sort()
    class_count = np.zeros(len(unique_classes) + 1, dtype=int)

    for x in classes:
        class_count[x] += 1

    class_percentage = class_count[1:] / nr_rows

    for x in unique_classes:
        print(f"Class {x}: {class_percentage[x - 1] : %}")

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

    print(f"\nMale percentage: {genders_percentage[1] : %}")
    print(f"Female percentage: {genders_percentage[0] : %}\n")

    # Realizam graficele pentru reprezentarea rezultatelor
    survived_array = [1 - survived_percentage, survived_percentage]
    plt.figure()
    plt.bar(["False", "True"], survived_array)
    plt.xticks(["False", "True"])
    plt.title("Survivors plot")
    plt.ylabel("Percentage")
    plt.xlabel("Survived")
    plt.savefig("task2-plots/Survivors.png")
    plt.show()

    plt.figure()
    plt.bar(unique_classes, class_percentage)
    plt.xticks(unique_classes)
    plt.title("Classes plot")
    plt.ylabel("Percentage")
    plt.xlabel("Classes")
    plt.savefig("task2-plots/Classes.png")
    plt.show()

    plt.figure()
    plt.bar(unique_genders, genders_percentage)
    plt.title("Genders plot")
    plt.ylabel("Percentage")
    plt.xlabel("Gender")
    plt.savefig('task2-plots/Male-Female.png')
    plt.show()

def task_3():
    print("\n-------------- TASK 3 --------------\n")
    print("Histograms added to task3-plots directory.\n")
    # Pentru fiecare coloana din dataframe, verificam daca aceasta contine
    # valori numerice (int sau float) si construim histogramele.
    for column in train_df:
        x = train_df[column]
        if x.dtype == int or x.dtype == float:
            plt.figure()
            plt.hist(x, bins = 20, edgecolor = 'black')
            plt.title(f"{column} histogram")
            plt.ylabel("Nr. People")
            plt.xlabel(column)
            plt.savefig(f"task3-plots/{column}")
            plt.show()

def task_4():
    print("\n-------------- TASK 4 --------------\n")
    # print("Valori lipsa => Proportie")
    for column in train_df:
        x = train_df[column]
        missing_values = x.isnull().sum()

        if missing_values != 0:
            survived_array = [0, 0]
            for i in range(len(x)):
                if x[i] != x[i]:
                    survived_array[train_df.at[i, 'Survived']] += 1

            print(f"{column}:\nMissing values: {missing_values} out of {nr_rows} => Proportion = {missing_values / nr_rows : .5f}")
            print(f"Did not survive = {survived_array[0]}, Survived = {survived_array[1]}\n")

def task_5():
    # Initializam o lista care va contine numarul de persoane din fiecare
    # categorie de varsta si adaugam coloana AgeCategory imediat dupa coloana Age.
    # Vom incadra persoanele a caror varsta este necunoscuta in categoria 0.
    category = [0, 0, 0, 0]
    ages = train_df['Age']
    train_df.insert(6, 'AgeCategory', 0)

    # Parcurgem coloana cu varstele, incrementam valoarea corespunzatoare din lista
    # de categorii si actualizam categoria de varsta a persoanei din tabel.
    for i in range(len(ages)):
        if ages[i] <= 20:
            category[0] += 1
            train_df.at[i, 'AgeCategory'] = 1
        elif ages[i] <= 40:
            category[1] += 1
            train_df.at[i, 'AgeCategory'] = 2
        elif ages[i] <= 60:
            category[2] += 1
            train_df.at[i, 'AgeCategory'] = 3
        elif ages[i] == ages[i]:
            category[3] += 1
            train_df.at[i, 'AgeCategory'] = 4

    # Realizam graficul pentru a evidentia rezultatele
    plt.figure()
    plt.title("Age categories")
    plt.ylabel("Number of passangers")
    plt.xlabel("Age category")
    plt.xticks([1, 2, 3, 4])
    plt.bar([1, 2, 3, 4], category)
    plt.savefig("task5-plots/AgeCategories.png")
    plt.show()


# Determinam ce task vrem sa rulam
# Daca nu se specifica un task, se vor executa toate
task = 0
if len(sys.argv) > 1:
    task = int(sys.argv[1])

train_df = pd.read_csv('input/train.csv')
nr_columns = len(train_df.axes[1])
nr_rows = len(train_df.axes[0])

if task == 1 or task == 0:
    task_1()

if task == 2 or task == 0:
    task_2()

if task == 3 or task == 0:
    task_3()

if task == 4 or task == 0:
    task_4()

if task == 5 or task == 0:
    task_5()