import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sbrn

class SP:

    def __init__(self):
        self.path = './../datasets/'
        self.datasetname = 'iris.dat'
        self.trainingSize = 0.8
        self.eta = 0.1
        self.realizations = 1
        self.epoch = 150
        self.dataset = np.asarray([])
        self.trainingSet = []
        self.testSet = []
        self.w = []
        self.allw = []
        self.hitratings = []
        self.confusionMatrixes = []


    def predict(self, u):
        return np.where(u >=0.0,1,0)


    def training(self):
        c = 0  # Epoch counter
        while c < self.epoch:
            ae = 0  # Accumulated Error
            for i in self.trainingSet:
                print(i)
                print(self.w)
                u = np.dot(self.w, i[0:len(i)-1])
                y = int(self.predict(u))
                e = i[len(i) - 1] - y
                self.w = self.adjustWeights(i[0:len(i)-1], e)
            #   ae += e
            #if ae == 0:
            #    break

            c += 1
            np.random.shuffle(self.trainingSet)
        self.allw.append(self.w)

    def test(self):
        cm = np.zeros([2, 2])
        c = 0
        for i in self.testSet:
            u = np.dot(self.w, i[0:len(i)-1])
            y = int(self.predict(u))
            e = i[len(i) - 1] - y

            if e == 0:
                c += 1
                if y == 1:
                    cm[1, 1] += 1
                else:
                    cm[0, 0] += 1
            else:
                if y == 1:
                    cm[1, 0] += 1
                else:
                    cm[0, 1] += 1

        hr = c/len(self.testSet)
        self.hitratings.append(hr)
        self.confusionMatrixes.append(cm)
        print('Hit Rating: ', hr*100)
        print(cm)

    def insertbias(self,dataset):
        new = []
        for i in range(len(dataset)):
            new.append(np.insert(dataset[i],0,-1))
            return np.asarray(new)


    def inserty(self,x,y):
        new = []
        for i in range(len(x)):
            new.append(np.insert(x[i],-1, y[i]))
        return np.asarray(new)


    def readData(self,path,datasetname):
        datasetlocation = path + datasetname
        data = np.array(pd.read_csv(datasetlocation, delimiter=',', header=None))
        return data

    def artificialgen(self):
        data = []
        for i in range(40):
            if(i < 10):
                x = 0 + random.uniform(-0.1, 0.1)
                y = 0 + random.uniform(-0.1, 0.1)
                data.append([x, y, 0])
            elif(i >= 10 and i < 20):
                x = 1 + random.uniform(-0.1, 0.1)
                y = 0 + random.uniform(-0.1, 0.1)
                data.append([x, y, 0])
            elif(i >= 20 and i < 30):
                x = 0 + random.uniform(-0.1, 0.1)
                y = 1 + random.uniform(-0.1, 0.1)
                data.append([x, y, 0])
            else:
                x = 1 + random.uniform(-0.1, 0.1)
                y = 1 + random.uniform(-0.1, 0.1)
                data.append([x, y, 1])
        return np.array(data)

    def classSelection(self,datasetoriginal, opt):
        dataset = datasetoriginal
        if opt == 1:  ##Setosa x All##
            for i in dataset:
                if i[len(i) - 1] == 'Iris-setosa':
                    i[len(i) - 1] = 1
                else:
                    i[len(i) - 1] = 0
        elif opt == 2:  #Versicolor x All#
            for i in dataset:
                if i[len(i) - 1] == 'Iris-versicolor':
                    i[len(i) - 1] = 1
                else:
                    i[len(i) - 1] = 0
        else:  # Virginica x All#
            for i in dataset:
                if i[len(i) - 1] == 'Iris-virginica':
                    i[len(i) - 1] = 1
                else:
                    i[len(i) - 1] = 0
        return dataset

    def dividedata(self,dataset):
        np.random.shuffle(dataset)
        self.trainingSet = dataset[0: int(len(dataset) * self.trainingSize)]
        self.testSet = dataset[int(len(dataset) * self.trainingSize):]

    def createWeights(self, length):
        return np.random.rand(1, length)

    def adjustWeights(self,dataset, e):
        return self.w + (self.eta) * (e) * dataset

    def normalize(self, dataset):
        dataset.transpose()
        for i in range(dataset.shape[1]):
            max_ = max(dataset[:, i])
            min_ = min(dataset[:, i])

            for j in range(dataset.shape[0]):
                dataset[j, i] = (dataset[j, i] - min_) / (max_ - min_)
        dataset.transpose()

    def datagen(self, opt):
        if opt == 1:
            return self.artificialgen()
        else:
            return self.readData(self.path, self.datasetname)

    def preprocesssing(self, dataset):
        self.w = self.createWeights(dataset.shape[1])
        self.normalize(dataset[0:len(dataset) - 1])
        dataset = self.insertbias(dataset)
        self.dataset = dataset

    def irisattchoice(self,dataset, opt):
        if opt == 0:
            return dataset
        elif opt == 1:
            return dataset[:, [1, 2, -1]]
        else:
            return dataset[:, [3, 4, -1]]

    def plotdata(self, dataset, xlabel='x', ylabely='y', title='Dataset'):

        data = dataset[:, [1, 2, -1]]
        print(data)

        df = pd.DataFrame(data, columns=[xlabel, ylabely, title])
        sbrn.lmplot(x='x', y='y', data=df, fit_reg=False, hue=title, markers=["o", "*"])
        plt.title(title)
        plt.show()


    def perceptron(self,dataset):
        #self.preprocesssing(dataset)
        self.w = self.createWeights(dataset.shape[1])
        self.normalize(dataset[0:len(dataset) - 1])
        dataset = self.insertbias(dataset)

        print(dataset)
        self.dataset = dataset

        for i in range(self.realizations):
            np.random.shuffle(dataset)
            self.dividedata(dataset)
            print(dataset)
            self.training()
            self.test()
            self.w = self.createWeights(self.w.shape[1]) #reset w for next iteraction
