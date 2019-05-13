import matplotlib.pyplot as plt
import numpy as np
from src import SP as sp


def plot_decision_region(dataset,xlabel, ylabely, title, weights, testdata):
    plt.rcParams['figure.figsize'] = (11, 7)
    plot_colors = "cb"
    plot_step = 0.005
    class_names = [0, 1]

    base = dataset

    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1

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
    print

    for i in range(len(dataset)):
        if dataset[i,-1] == 0:
            plt.plot(dataset[i][1], dataset[i][2], "b^")
        else:
            plt.plot(dataset[i][1], dataset[i][2], "ro")

    for i in range(len(testdata)):
        if testdata[i, -1] == 0:
            plt.plot(testdata[i][1], testdata[i][2], "w^")
        else:
            plt.plot(testdata[i][1], testdata[i][2], "ko")

    plt.show()
