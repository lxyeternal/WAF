# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 14:35
# @Author  : blue
# @FileName: svm.py
# @Software: PyCharm


import joblib
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from evaluate import plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier


class SVMModel:
    def __init__(self,label_list,train_data,train_label,test_data,test_label):
        self.label_list = label_list
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label


    def svm_alg(self):
        random_state = np.random.RandomState(10)
        model = OneVsRestClassifier(
            svm.SVC(kernel='linear', max_iter=1000, random_state=random_state))  # 线性核函数：linear   多项式核函数：poly    高斯核函数：rbf     sigmod核函数：sigmoid
        print("[INFO] Successfully initialize a SVM model !")
        print("[INFO] Training the model…… ")
        #   模型训练
        clt = model.fit(self.train_data,self.train_label)
        print("[INFO] Model training completed !")
        # 保存训练好的模型，下次使用时直接加载就可以了
        model_path = '../model/mult_svm.pkl'
        joblib.dump(clt,model_path)
        print("[INFO] Model has been saved !")
        y_test_pred = clt.predict(self.test_data)
        print(y_test_pred)
        ov_acc = metrics.accuracy_score(y_test_pred, self.test_label)
        print("overall accuracy: %f" % (ov_acc))
        svm_predictions = y_test_pred.tolist()
        predictios = list()
        for i in svm_predictions:
            try:
                predictios.append(i.index(1))
            except:
                predictios.append(0)
        real_label = list()
        self.test_label = self.test_label.tolist()
        for i in self.test_label:
            real_label.append(i.index(1))
        #   计算模型的准确率
        svm_acc = metrics.accuracy_score(predictios, real_label)
        svm_confusion_matrix = metrics.confusion_matrix(real_label, predictios, sample_weight=None)
        print("confusion metrix:\n", svm_confusion_matrix)
        print("overall accuracy: %f" % (svm_acc))
        svm_classification_rep = classification_report(real_label, predictios, target_names=self.label_list)
        print("classification report: \n", svm_classification_rep)
        #  绘制混淆矩阵
        plot_confusion_matrix('SVM', svm_confusion_matrix, self.label_list)