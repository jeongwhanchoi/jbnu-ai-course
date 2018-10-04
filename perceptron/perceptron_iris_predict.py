import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_module.plotdregion import plot_decision_region
from ml_module.perceptron import Perceptron

df = pd.read_csv('../input/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values



def TrainPerceptronModel(X, y):
    ppn = Perceptron(eta=0.1, n_iter=10)

    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')

    plt.tight_layout()
    # plt.savefig('./perceptron_1.png', dpi=300)
    plt.show()

    plot_decision_region(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('./perceptron_2.png', dpi=300)
    plt.show()

    print(ppn.errors_)

    return ppn


def getNewX():
    n = 2
    newX = []
    for i in range(0, n):
        test = input("Enter the sepal & petal length you want to test\n")
        if (test.isdigit()):
            newX.insert(i, float(test))  # statement1
        else:
            newX.insert(i, test)  # statement2
    print('----------------------\nThe new data:', newX)

    return newX


def my_test():
    newX = getNewX()

    result = ppn.predict(newX)
    print('\n----------------------')
    if result == 1:
        print('Your input data is classified as VERISCOLOR')
    else:
        print('Your input data is classified as SETOSA')

ppn = TrainPerceptronModel(X, y)

my_test()