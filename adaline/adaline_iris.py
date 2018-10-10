import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_module.adaline import AdalineGD

if __name__ == '__main__' :
    df = pd.read_csv('../input/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y=='Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    adal = AdalineGD(eta=0.01, n_iter=10).fit(X, y)
    # ax[0].plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_), marker='o')
    ax[0].plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(SQE)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    adal2 = AdalineGD(eta=0.0001, n_iter=10).fit(X, y)
    ax[1].plot(range(1, len(adal2.cost_) + 1), np.log10(adal2.cost_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(SQE)')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.5, hspace=0.5)
    plt.show()