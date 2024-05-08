# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 14:35
# @Author  : blue
# @FileName: gru.py
# @Software: PyCharm


import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
from evaluate import plot_confusion_matrix, evaluate_from_confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Dense, GRU


#  categorical_crossentropy  标签为one_hot编码
#  binary_crossentropy      单标签情况

class GRUModel:
    def __init__(self,label_list,train_data,train_label,test_data,test_label):
        self.label_list = label_list
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.VALIDATION_SPLIT = 0.3
        self.nb_lstm_outputs = 200
        self.labels_index = 3
        self.batch_size = 64
        self.epochs = 20
        self.train_data = self.train_data.astype(np.float64)
        self.test_data = self.test_data.astype(np.float64)
        self.train_data = self.train_data.reshape((self.train_data.shape[0], 1, self.train_data.shape[1]))
        self.test_data = self.test_data.reshape((self.test_data.shape[0], 1, self.test_data.shape[1]))


    def gru_alg(self):
        model = Sequential()
        model.add(GRU(units=self.nb_lstm_outputs,return_sequences=False,dropout=0.2,recurrent_dropout=0.2))
        model.add(BatchNormalization())
        model.add(Dense(self.labels_index, activation='softmax'))
        # model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print("[INFO] Successfully initialize a GRU model !")
        print("[INFO] Training the model…… ")
        model.fit(self.train_data, self.train_label, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.VALIDATION_SPLIT,verbose=1)
        print("[INFO] Model training completed !")
        model.save('../model/gru.h5')
        score, acc = model.evaluate(self.test_data, self.test_label)
        print("overall accuracy: %f" % acc)
        gru_predictions = model.predict(self.test_data)
        print(gru_predictions)
        gru_predictions = gru_predictions.tolist()
        predictios = list()
        for i in gru_predictions:
            try:
                predictios.append(i.index(max(i)))
            except:
                predictios.append(0)
        real_label = list()
        self.test_label = self.test_label.tolist()
        for i in self.test_label:
            real_label.append(i.index(1))
        #   计算模型的准确率
        gru_acc = metrics.accuracy_score(predictios, real_label)
        gru_confusion_matrix = metrics.confusion_matrix(real_label, predictios, sample_weight=None)
        print("confusion metrix:\n", gru_confusion_matrix)
        print("overall accuracy: %f" % (gru_acc))
        gru_classification_rep = classification_report(real_label, predictios, target_names=self.label_list)
        print("classification report: \n", gru_classification_rep)
        #  绘制混淆矩阵
        plot_confusion_matrix('GRU', gru_confusion_matrix, self.label_list)
        evaluate_metrics = evaluate_from_confusion_matrix(gru_confusion_matrix)
        print("Classification Metrics:")
        print("Accuracy: {:.2f}%".format(evaluate_metrics['Accuracy'] * 100))
        print("Macro Precision: {:.2f}%".format(evaluate_metrics['Macro Precision'] * 100))
        print("Macro Recall: {:.2f}%".format(evaluate_metrics['Macro Recall'] * 100))
        print("Macro F1 Score: {:.2f}%".format(evaluate_metrics['Macro F1 Score'] * 100))