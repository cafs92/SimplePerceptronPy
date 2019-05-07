import src.SP as sp
import numpy as np

def main():
    PS = sp.SP()
    #OPT1 = Artificial x Iris
    #OPT2 = 1: Setosa x All
    #       2: Versicolor x All
    #       3: Virginica x All
    #OPT3 = Iris train 2 att:
    #       1: yes
    #       2: no
    dataset = np.array(PS.artificialgen())

    PS.perceptron(dataset)



if __name__ == '__main__':
    main()