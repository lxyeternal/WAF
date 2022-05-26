# -*- coding: utf-8 -*-
# @Project ：AiWaf
# @Time    : 2022/5/25 22:49
# @Author  : honywen
# @FileName: predict.py
# @Software: PyCharm


import joblib
import numpy as np
from tensorflow import keras
from vecmodel import payload2vec


def translabel(pred):
    pred = pred.tolist()[0]
    if pred == [0,0,0]:
        label = "正常"
    elif pred == [1,0,0]:
        label = "正常"
    elif pred == [0,1,0]:
        label = "XSS攻击"
    else:
        label = "SQL注入攻击"
    return label


def cnn_translabel(pred):
    pred = pred.tolist()[0]
    label_index = pred.index(max(pred))
    label_list = ["正常","XSS攻击","SQL注入攻击"]
    return label_list[label_index]


def predict(payload):
    #  加载预测模型
    rf_model = joblib.load("../model/mult_rf.pkl")
    knn_model = joblib.load("../model/mult_knn.pkl")
    cnn_model = keras.models.load_model('../model/mult_cnn')
    gru_model = keras.models.load_model('../model/gru.h5')
    svm_model = joblib.load('../model/mult_svm.pkl')

    #  payload进行预处理
    payload_vec = payload2vec(payload)
    payload_vec = np.array(payload_vec)
    payload_reshape = payload_vec.reshape((1, 1, payload_vec.shape[1], 1))
    payload_reshape_gru = payload_vec.reshape((1, 1, payload_vec.shape[1]))
    rf_pred = rf_model.predict(payload_vec)
    knn_pred = knn_model.predict(payload_vec)
    svm_pred = svm_model.predict(payload_vec)
    cnn_pred = cnn_model.predict(payload_reshape)
    gru_pred = gru_model.predict(payload_reshape_gru)

    rf_label = translabel(rf_pred)
    knn_label = translabel(knn_pred)
    cnn_label = cnn_translabel(cnn_pred)
    gru_label = cnn_translabel(gru_pred)
    svm_label = translabel(svm_pred)

    print("RandomForest模型预测结果：" + rf_label)
    print("KNN模型预测结果：" + knn_label)
    print("CNN模型预测结果：" + cnn_label)
    print("GRU模型预测结果：" + gru_label)
    print("SVM模型预测结果：" + svm_label)


predict("http://honywen.com/examples/jsp/checkbox/bandwidth/index.cgi?action=showmonth&year=<script>foo</script>&month=<script>foo</script>")



