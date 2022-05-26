# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:47
# @Author  : blue
# @FileName: knn.py
# @Software: PyCharm


import joblib
from sklearn.metrics import classification_report
from sklearn import metrics
from evaluate import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


class KNNModel:

    def __init__(self,label_list,train_data,train_label,test_data,test_label):
        self.label_list = label_list
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label


    def knn_alg(self):
        kNN_classifier = KNeighborsClassifier(n_neighbors=1)
        print("[INFO] Successfully initialize a KNN model !")
        print("[INFO] Training the model…… ")
        kNN_classifier.fit(self.train_data,self.train_label)
        model_path = '../model/mult_knn.pkl'
        joblib.dump(kNN_classifier, model_path)
        target_predict = kNN_classifier.predict(self.test_data)
        score = kNN_classifier.score(self.test_data, self.test_label, sample_weight=None)
        knn_predictions = target_predict.tolist()
        predictios = list()
        for i in knn_predictions:
            try:
                predictios.append(i.index(1))
            except:
                predictios.append(0)
        real_label = list()
        self.test_label = self.test_label.tolist()
        for i in self.test_label:
            real_label.append(i.index(1))
        #   计算模型的准确率
        knn_acc = metrics.accuracy_score(predictios, real_label)
        knn_confusion_matrix = metrics.confusion_matrix(real_label, predictios, sample_weight=None)
        print("confusion metrix:\n", knn_confusion_matrix)
        print("overall accuracy: %f" % (knn_acc))
        knn_classification_rep = classification_report(real_label, predictios, target_names=self.label_list)
        print("classification report: \n", knn_classification_rep)
        #  绘制混淆矩阵
        plot_confusion_matrix('KNN', knn_confusion_matrix, self.label_list)

