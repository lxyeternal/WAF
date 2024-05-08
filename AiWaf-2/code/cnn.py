# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 11:20
# @Author  : blue
# @FileName: cnn.py
# @Software: PyCharm

from sklearn.metrics import classification_report
from sklearn import metrics
from evaluate import plot_confusion_matrix, evaluate_from_confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D


class CNNModel:

    def __init__(self,label_list,train_data,train_label,test_data,test_label):
        self.label_list = label_list
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.first_filters = 256
        self.second_filters = 128
        self.third_filters = 64
        self.kernel_size = (1, 3)
        self.pool_size = (1, 2)
        self.input_shape = (1, 200, 1)
        self.batch_size = 128
        self.epochs = 1
        self.train_data = self.train_data.reshape((self.train_data.shape[0], 1, self.train_data.shape[1], 1))
        self.test_data = self.test_data.reshape((self.test_data.shape[0], 1, self.test_data.shape[1], 1))

    def cnn_alg(self):
        model = Sequential()
        model.add(Convolution2D(self.first_filters, self.kernel_size, padding='valid', strides=1,
                                input_shape=self.input_shape))  # 卷积层1
        model.add(Convolution2D(self.first_filters, self.kernel_size, padding='valid', strides=1))  # 卷积层2
        model.add(MaxPooling2D(pool_size=self.pool_size))  # 池化层1

        model.add(Convolution2D(self.second_filters, self.kernel_size, padding='valid', strides=1))  # 卷积层3
        model.add(Convolution2D(self.second_filters, self.kernel_size, padding='same', strides=1))  # 卷积层4
        model.add(Convolution2D(self.second_filters, self.kernel_size, padding='same', strides=1))  # 卷积层5
        model.add(MaxPooling2D(pool_size=self.pool_size))  # 池化层2

        model.add(Convolution2D(self.third_filters, self.kernel_size, padding='same', strides=1))  # 卷积层6
        model.add(Convolution2D(self.third_filters, self.kernel_size, padding='same', strides=1))  # 卷积层7
        model.add(Convolution2D(self.third_filters, self.kernel_size, padding='same', strides=1))  # 卷积层8
        model.add(MaxPooling2D(pool_size=self.pool_size))  # 池化层3
        model.add(Flatten())  # 拉成一维数据
        model.add(Dense(100))  # 全连接层1
        model.add(Dropout(0.5))  # 随机失活
        model.add(Dense(20))  # 全连接层2
        model.add(Dense(3, activation='softmax'))
        model.summary()

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        # 模型训练
        print("[INFO] Successfully initialize a CNN model !")
        print("[INFO] Training the model…… ")
        model.fit(self.train_data, self.train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        #  保存训练好的模型
        model_path = '../model/mult_cnn'
        model.save(model_path)
        # 评估模型
        cnn_predictions = model.predict(self.test_data)
        # score = model.evaluate(self.test_data, self.test_label, verbose=0)
        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])
        print(cnn_predictions)
        cnn_predictions = cnn_predictions.tolist()
        predictios = list()
        for i in cnn_predictions:
            try:
                predictios.append(i.index(max(i)))
            except:
                predictios.append(0)
        real_label = list()
        self.test_label = self.test_label.tolist()
        for i in self.test_label:
            real_label.append(i.index(1))
        #   计算模型的准确率
        cnn_acc = metrics.accuracy_score(predictios, real_label)
        cnn_confusion_matrix = metrics.confusion_matrix(real_label, predictios, sample_weight=None)
        print("confusion metrix:\n", cnn_confusion_matrix)
        print("overall accuracy: %f" % (cnn_acc))
        cnn_classification_rep = classification_report(real_label, predictios, target_names=self.label_list)
        print("classification report: \n", cnn_classification_rep)
        #  绘制混淆矩阵
        plot_confusion_matrix('CNN', cnn_confusion_matrix, self.label_list)
        evaluate_metrics = evaluate_from_confusion_matrix(cnn_confusion_matrix)
        print("Classification Metrics:")
        print("Accuracy: {:.2f}%".format(evaluate_metrics['Accuracy'] * 100))
        print("Macro Precision: {:.2f}%".format(evaluate_metrics['Macro Precision'] * 100))
        print("Macro Recall: {:.2f}%".format(evaluate_metrics['Macro Recall'] * 100))
        print("Macro F1 Score: {:.2f}%".format(evaluate_metrics['Macro F1 Score'] * 100))

