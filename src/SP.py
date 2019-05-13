import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from src import util as ut

class SP:
    def __init__(self):
        self.path = './../datasets/'
        self.datasetname = 'iris.dat'
        self.trainingSize = 0.8
        self.eta = 0.1
        self.realizations = 20
        self.epoch = 150
        self.dataset = np.asarray([])
        self.trainingSet = []
        self.testSet = []
        self.w = []
        self.hitratings = []
        self.confusionMatrixes = []
        self.min = 100
        self.max = 0
        self.allw = []
        self.alltrainings = []
        self.alltests = []
        self.x = ''
        self.y = ''
        self.title = ''


    def predict(self, u):
        return np.where(u >=0.0,1,0)


    def training(self):
        c = 0  # Epoch counter
        while c < self.epoch:
            ae = 0  # Accumulated Error
            for i in self.trainingSet:
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
        w = np.random.rand(1, length)[0]
        return w

    def adjustWeights(self,dataset, e):
        return self.w + (self.eta) * (e) * dataset

    def normalize(self, dataset):
        dataset.transpose()
        for i in range(dataset.shape[1] -1):
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

    def dataselect(self, opt):
        if opt == 0:
            return self.readData(self.path, self.datasetname)
        elif opt == 1:
            dataset = self.readData(self.path, self.datasetname)
            return dataset[:, [0, 1, -1]]
        elif opt == 2:
            dataset = self.readData(self.path, self.datasetname)
            return dataset[:, [2, 3, -1]]
        else:
            return self.artificialgen()

    def postprocessing(self, opt1,opt2):
        #auxmin = 1000
        auxmax = -1
        for i in range(len(self.hitratings)):
           # if self.hitratings[i] <= auxmin:
           #     self.min = i
           #     auxmin = self.hitratings[i]
            if self.hitratings[i] >= auxmax:
                self.max = i
                auxmax = self.hitratings[i]
        self.setplotatt(opt1,opt2)

        if opt1 != 0:
         #   print('\n\n\nPloting data for minimum hit rating: ')
         #   ut.plot_decision_region(self.dataset, self.x, self.y,
         #                           self.title +' Min', self.allw[self.min], self.alltrainings[self.min])
            print('\n\n\nPloting data for maximum hit rating: ')
            ut.plot_decision_region(self.dataset, self.x, self.y,
                                    self.title +' Max', self.allw[self.max], self.alltrainings[self.max])


    def setplotatt(self,opt1,opt2):
        if opt1 == 1:
            self.x = 'Sepal Length'
            self.y = 'Sepal Width'
            if opt2 == 1:
                self.title = 'Iris Dataset - Setosa x All'
            elif opt2 == 1:
                self.title = 'Iris Dataset - Versicolor x All'
            else:
                self.title = 'Iris Dataset - Virginica x All'
        elif opt1 == 2:
            self.x = 'Petal Length'
            self.y = 'Petal Width'
            if opt2 == 1:
                self.title = 'Iris Dataset - Setosa x All'
            elif opt2 == 1:
                self.title = 'Iris Dataset - Versicolor x All'
            else:
                self.title = 'Iris Dataset - Virginica x All'
        elif opt1 == 3:
            self.x = 'x'
            self.y = 'y'
            self.title = 'Artificial Dataset'

    def perceptron(self, opt, opt2):
        dataset = self.dataselect(opt)
        print(dataset[1])
        if opt != 3:
            dataset = self.classSelection(dataset, opt2)

        self.w = self.createWeights(dataset.shape[1])
        self.normalize(dataset[:len(dataset)])
        dataset = self.insertbias(dataset)
        self.dataset = dataset

        for i in range(self.realizations):
            self.w = self.createWeights((self.dataset.shape[1]) - 1)
            np.random.shuffle(dataset)
            self.dividedata(dataset)
            self.alltrainings.append(self.trainingSet)
            self.alltests.append(self.trainingSet)
            self.training()
            self.allw.append(self.w)
            self.test()

        print('Accuracy: ', np.asarray(self.hitratings).mean())
        print('Standard Deviation: ',np.asarray(self.hitratings).std())
        self.postprocessing(opt,opt2)
