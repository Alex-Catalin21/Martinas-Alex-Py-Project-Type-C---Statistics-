import random
from math import sqrt
import csv
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import sklearn
import matplotlib.pyplot as plt
path = "C:\\Users\\Alex\\Desktop\\GitHub\\Martinas-Alex-Py-Project-Type-C---Statistics-\\data.csv"

def Pearson(X, Y, n):
    sum_X = 0
    sum_Y = 0
    sum_XY = 0
    squareSum_X = 0
    squareSum_Y = 0

    i = 0
    while i < n:
        sum_X = sum_X + X[i]
        sum_Y = sum_Y + Y[i]
        sum_XY = sum_XY + X[i] * Y[i]

        squareSum_X = squareSum_X + X[i] * X[i]
        squareSum_Y = squareSum_Y + Y[i] * Y[i]

        i = i + 1

    corr = (float)(n * sum_XY - sum_X * sum_Y)/(float)(sqrt((n * squareSum_X - sum_X * sum_X)*(n* squareSum_Y - sum_Y * sum_Y)))
    return corr

def generate():
    age = [i for i in range(10, 100)]
    gen = ['F', 'M']
    iq = [i for i in range(85, 115)]
    nationality = ["Romanian", "Japanese", "German", "French", "British", "Austrian", "American", "Taiwanese"]

    try:
        f = open("data.csv", "w")
        try:
            f.write("Age" + "," + "Gen" + "," + "Iq" + "," + "Nationality" + "\n")
            for i in range(1, 30):
                f.write(str(random.choice(age)) + ',' + random.choice(gen) + ',' + str(random.choice(iq)) + ',' + random.choice(nationality) + '\n')
        except:
            print("Writting error")
        finally:
            f.close()
    except:
        print("Opening error")


def statistics(path):
    generate()
    f = open("data.csv", "r")
    file = csv.DictReader(f)
    age = []
    gen = []
    iq = []
    nationality = []

    for col in file:
        age.append(col['Age'])
        gen.append(col['Gen'])
        iq.append(col['Iq'])
        nationality.append(col['Nationality'])

    print(age)
    age = [int(x) for x in age]
    iq = [int(x) for x in iq]
    #mean
    mean_age = sum(age)/len(age)
    mean_iq = sum(iq)/len(iq)
    print("Mean is:", mean_age)
    #median
    age2 = age.copy()
    age2.sort()
    mid = len(age2) // 2
    res = (age2[mid] + age2[~mid]) / 2
    print("Median:", str(res))
    #min
    print("Minimum is:", min(age))
    #max
    print("Maximum is:", max(age))
    #stddev sqrt(sum(X - mean)**2 / nr observations
    summ = 0
    for i in age2:
        summ += (i-mean_age)**2
    stddev = sqrt(summ/(len(age2)-1))
    print("stddev py:", stddev)
    #pearson
    n = len(age)
    print("Pearson Py:", Pearson(age, iq, n))

    #cov
    diff_age = [age[i]-mean_age for i in range(len(age))]
    diff_iq = [iq[i]-mean_iq for i in range(len(iq))]
    sum_ai = [diff_age[i]*diff_iq[i] for i in range(len(age))]
    cov = sum(sum_ai)/len(sum_ai)
    print("covariance py:", cov)
    #quartiles..
    Q1 = (len(age2) + 1) * 1//4
    Q2 = (len(age2) + 1) * 2//4
    Q3 = (len(age2) + 1) * 3//4
    print("Py Quartiles prost:", age2[int(Q1)], age2[int(Q2)], age2[int(Q3)])
    #statistics
    print("Quartiles np:", np.quantile(age2, [0.25, 0.5, 0.75]) )
    #plot Pearson
    x = pd.Series(age)
    y = pd.Series(iq)
    correlation = y.corr(x)
    print("corr:", correlation)
    plt.title('Correlation')
    plt.scatter(x, y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))
    (np.unique(x)), color='red')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()
def statistics2(path):
    df = pd.read_csv('data.csv')
    mean1 = df['Age'].mean()
    print("Library mean Age:", str(mean1))
    max1 = df['Age'].max()
    min1 = df['Age'].min()
    std1 = df['Age'].std()
    median1 = df['Age'].median()
    print("Library Max Age:", str(max1))
    print("Library Min Age:", str(min1))
    print("Library std Age:", str(std1))
    print("Library median Age:", str(median1))
    # pearson
    list1 = df['Age']
    list2 = df['Iq']
    corr, _ = pearsonr(list1, list2)
    print("Library Pearson Age-Iq:", corr)
    print("Library Covariance Age-Iq:", df.Age.cov(df.Iq))

statistics(path)
statistics2(path)
