# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 14:35
# @Author  : blue
# @FileName: rf.py
# @Software: PyCharm


import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from evaluate import plot_confusion_matrix


class RFModel:
    def __init__(self,label_list,train_data,train_label,test_data,test_label):
        self.label_list = label_list
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label


    def rf_alg(self):
        rf_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                          max_depth=None, max_features='auto', max_leaf_nodes=None,
                                          min_samples_leaf=1, min_samples_split=2,
                                          min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                          oob_score=False, random_state=None, verbose=0,
                                          warm_start=False)

        print("[INFO] Successfully initialize a RF model !")
        print("[INFO] Training the model…… ")
        #   进行模型训练
        rf_model.fit(self.train_data, self.train_label)
        print("[INFO] Model training completed !")
        # 保存训练好的模型，下次使用时直接加载就可以了
        model_path = '../model/mult_rf.pkl'
        joblib.dump(rf_model, model_path)
        print("[INFO] Model has been saved !")
        #   进行测试数据的预测
        rf_predictions = rf_model.predict(self.test_data)
        #   给出每一个标签的预测概率
        rf_predictions = rf_predictions.tolist()
        predictios = list()
        for i in rf_predictions:
            try:
                predictios.append(i.index(1))
            except:
                predictios.append(0)
        real_label = list()
        self.test_label = self.test_label.tolist()
        for i in self.test_label:
            real_label.append(i.index(1))
        #   计算模型的准确率
        rf_acc = metrics.accuracy_score(rf_predictions, real_label)
        rf_confusion_matrix = metrics.confusion_matrix(real_label, predictios, sample_weight=None)
        print("confusion metrix:\n", rf_confusion_matrix)
        print("overall accuracy: %f" % (rf_acc))
        rf_classification_rep = classification_report(real_label, predictios, target_names=self.label_list)
        print("classification report: \n", rf_classification_rep)
        #  绘制混淆矩阵
        plot_confusion_matrix('Random Forest',rf_confusion_matrix, self.label_list)