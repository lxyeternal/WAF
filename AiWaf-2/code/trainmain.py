# -*- coding: utf-8 -*-
# @Project ：AiWaf
# @Time    : 2022/5/25 19:50
# @Author  : honywen
# @FileName: trainmain.py
# @Software: PyCharm



import numpy as np
from cnn import CNNModel
from knn import KNNModel
from svm import SVMModel
from rf import RFModel
from gru import GRUModel
from splitdata import splitmain


class ALG:

    def __init__(self):

        self.label_list = ['0', '1', '2']
        self.train_data = list()
        self.test_data = list()
        self.validation_data = list()
        self.train_label = list()
        self.test_label = list()
        self.validation_label = list()

    def Data(self):
        train_data,train_label,validation_data,validation_label,test_data,test_label = splitmain()
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.validation_data = validation_data
        self.train_data = np.array(train_data)
        self.train_label = np.array(train_label)
        self.test_data = np.array(test_data)
        self.test_label = np.array(test_label)
        self.validation_data = np.array(validation_data)
        self.validation_label = np.array(validation_label)


    def KNN(self):
        knnmodel = KNNModel(self.label_list, self.train_data,self.train_label,self.test_data, self.test_label)
        knnmodel.knn_alg()

    def CNN(self):
        cnnmodel = CNNModel(self.label_list, self.train_data,self.train_label,self.test_data,self.test_label)
        cnnmodel.cnn_alg()

    def SVM(self):
        svmmodel = SVMModel(self.label_list, self.train_data,self.train_label,self.test_data,self.test_label)
        svmmodel.svm_alg()


    def RF(self):
        rfmodel = RFModel(self.label_list, self.train_data, self.train_label, self.test_data,self.test_label)
        rfmodel.rf_alg()


    def GRU(self):
        grummodel = GRUModel(self.label_list, self.train_data, self.train_label, self.test_data,self.test_label)
        grummodel.gru_alg()



if __name__ == '__main__':

    alg = ALG()
    alg.Data()
    alg.RF()
    alg.KNN()
    alg.SVM()
    alg.CNN()
    alg.GRU()