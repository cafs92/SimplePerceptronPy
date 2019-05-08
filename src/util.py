import matplotlib.pyplot as plt
import numpy as np
from src import SP as sp


def plot_decision_region(dataset,xlabel, ylabely, title, weights):
    plt.rcParams['figure.figsize'] = (11, 7)
    plot_colors = "cb"
    plot_step = 0.02
    class_names = [0, 1]

    base = dataset

    x_min, x_max = base[:, 1].min() - 1, base[:, 1].max() + 1
    y_min, y_max = base[:, 2].min() - 1, base[:, 2].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # plt.tight_layout()
    data = np.c_[xx.ravel(), yy.ravel()]
    bias_col = -np.ones(data.shape[0])

    data = np.insert(data, 0, bias_col, axis=1)
    ps = sp.SP()
    ps.w = weights

    u = np.dot(ps.w, data.transpose())
    Z = np.array([ps.predict(i) for i in u])
    Z = Z.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    y = base[:, -1]

    plt.xlabel(xlabel)
    plt.ylabel(ylabely)
    plt.title(title)

    for i, color in zip(range(len(class_names)), plot_colors):
        idx = np.where(y == i)
        plt.scatter(base[idx, 1], base[idx, 2], c=color, label=class_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
    plt.show()
