import numpy as np
from ml_module.perceptron import Perceptron

if __name__ =='__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    ppn = Perceptron(eta=0.1)
    ppn.fit(X, y)
    print(ppn.errors_)
