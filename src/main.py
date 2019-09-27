import src.SP as sp
import numpy as np

def main():
    PS = sp.SP()
    #OPT1 = Artificial x Iris
    #       0: Iris full att
    #       1: Iris Sepal x Sepal
    #       2: Iris Petal x Petal
    #       3: Artificial AND
    #OPT2 = 1: Setosa x All
    #       2: Versicolor x All
    #       3: Virginica x All

    PS.perceptron(4, 1)
if __name__ == '__main__':
    main()