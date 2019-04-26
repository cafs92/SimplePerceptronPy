import math
import random as rng
import numpy as np

class DataManipulation():
    def __init__(self):
        self.dataSet = []
        self.trSet = []
        self.testSet = []
        self.percent = 0.8
        self.artificial()
        print(self.dataSet)



    def artificial(self):
        for i in range(40):
            if (i < 10):
                x = 0 + rng.uniform(-0.1, 0.1)
                y = 0 + rng.uniform(-0.1, 0.1)
                self.dataSet.append([x, y, 0])
            elif (i >= 10 and i < 20):
                x = 0 + rng.uniform(-0.1, 0.1)
                y = 1 + rng.uniform(-0.1, 0.1)
                self.dataSet.append([x, y, 0])
            elif (i >= 10 and i < 20):
                x = 1 + rng.uniform(-0.1, 0.1)
                y = 0 + rng.uniform(-0.1, 0.1)
                self.dataSet.append([x, y, 0])
            else:
                x = 1 + rng.uniform(-0.1, 0.1)
                y = 1 + rng.uniform(-0.1, 0.1)
                self.dataSet.append([x, y, 1])


    def readDataset(self):
        self.dataSet = np.loadtxt('./../iris.dat')
