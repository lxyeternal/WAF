# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 14:35
# @Author  : blue
# @FileName: rf.py
# @Software: PyCharm


import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from evaluate import plot_confusion_matrix, evaluate_from_confusion_matrix


class RFModel:
    def __init__(self, label_list, train_data, train_label, test_data, test_label):
        self.label_list = label_list
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

    def rf_alg(self):
        rf_model = RandomForestClassifier(n_estimators=10, max_features='sqrt', random_state=None)
        print("[INFO] Successfully initialize a RF model!")
        print("[INFO] Training the model…")
        rf_model.fit(self.train_data, np.argmax(self.train_label, axis=1))  # Assuming train_label is also one-hot
        print("[INFO] Model training completed!")

        # 保存模型
        model_path = '../model/mult_rf.pkl'
        joblib.dump(rf_model, model_path)
        print("[INFO] Model has been saved!")

        # 进行测试数据的预测
        rf_predictions = rf_model.predict(self.test_data)
        # 将预测的类别索引转换为one-hot编码以匹配test_label
        rf_predictions_onehot = np.eye(len(self.label_list))[rf_predictions]

        # 计算模型的准确率
        rf_acc = accuracy_score(np.argmax(self.test_label, axis=1), rf_predictions)  # 使用整数索引进行比较
        print("Overall accuracy: %f" % rf_acc)

        # 生成混淆矩阵和分类报告
        rf_confusion_matrix = confusion_matrix(np.argmax(self.test_label, axis=1), rf_predictions)
        print("Confusion matrix:\n", rf_confusion_matrix)
        rf_classification_rep = classification_report(np.argmax(self.test_label, axis=1), rf_predictions, target_names=self.label_list)
        print("Classification report: \n", rf_classification_rep)

        # 绘制混淆矩阵
        plot_confusion_matrix('Random Forest', rf_confusion_matrix, self.label_list)
        evaluate_metrics = evaluate_from_confusion_matrix(rf_confusion_matrix)
        print("Classification Metrics:")
        print("Accuracy: {:.2f}%".format(evaluate_metrics['Accuracy'] * 100))
        print("Macro Precision: {:.2f}%".format(evaluate_metrics['Macro Precision'] * 100))
        print("Macro Recall: {:.2f}%".format(evaluate_metrics['Macro Recall'] * 100))
        print("Macro F1 Score: {:.2f}%".format(evaluate_metrics['Macro F1 Score'] * 100))
