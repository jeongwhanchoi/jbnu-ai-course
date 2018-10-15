import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('r', 'b', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[: len(np. unique(y))])

    # decision surface  그리기
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() +1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() +1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z. reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np. unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', linewidth=1, marker='o',
                    s=80, label='Test Set')

    plt.xlabel('Standardized Petal Length')
    plt.ylabel('Standardized Petal Width')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()
