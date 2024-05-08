# -*- coding: utf-8 -*-
# @Project ：AiWaf
# @Time    : 2022/5/25 20:57
# @Author  : honywen
# @FileName: evaluate.py
# @Software: PyCharm


import numpy as np
import itertools
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 绘制混淆矩阵
def plot_confusion_matrix(algname, cm, classes, normalized=True, cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalized : True:显示百分比, False:显示个数
    """
    title = algname + ' Confusion matrix'
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={
    'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalized else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../images/confusion_matrix_' + algname +'.png')
    # plt.show()
    plt.clf()


def draw_trainaccloss(algname,loss_list,acc_list):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()

    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("train loss")
    par1.set_ylabel("train accuracy")

    # plot curves
    p1, = host.plot(range(len(loss_list)), loss_list, label="loss")
    p2, = par1.plot(range(len(acc_list)), acc_list, label="accuracy")

    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.grid()
    plt.draw()
    plt.savefig('../images/trainaccloss_' + algname + '.jpg')
    # plt.show()
    plt.clf()


def model_evaluation(test_real_label,test_pred_label):
    sklearn_accuracy = accuracy_score(test_real_label, test_pred_label)
    sklearn_precision = precision_score(test_real_label, test_pred_label, average='weighted')
    sklearn_recall = recall_score(test_real_label, test_pred_label, average='weighted')
    sklearn_f1 = f1_score(test_real_label, test_pred_label, average='weighted')
    print(sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1)
    return sklearn_precision,sklearn_recall,sklearn_f1


def get_confusion_matrix(test_real_label,test_pred_label):
    label_list = ["0","1","2"]
    conf_matrix = confusion_matrix(test_real_label, test_pred_label, label_list)
    return conf_matrix


def draw_roc(test_real_label,test_pred_label,algname):
    fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(test_real_label, test_pred_label)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])

    plt.plot(fpr_keras, tpr_keras, label=' (auc = {:.4f})'.format(auc_keras))
    plt.title('ROC curve')
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('../images/roc_' + algname + '.jpg')
    # plt.show()
    plt.clf()


def evaluate_from_confusion_matrix(conf_matrix):
    """
    计算并返回基于混淆矩阵的分类评价指标。
    参数:
    conf_matrix (numpy.ndarray): 混淆矩阵，每一行表示真实类别，每一列表示预测类别。
    返回:
    dict: 包含准确率、宏观精确度、宏观召回率和宏观F1分数的字典。
    """
    # 准确率
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    
    # 精确度 (防止分母为0的情况)
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    precision = np.nan_to_num(precision)  # 将NaN转换为0
    macro_precision = np.mean(precision)
    
    # 召回率 (防止分母为0的情况)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    recall = np.nan_to_num(recall)  # 将NaN转换为0
    macro_recall = np.mean(recall)
    
    # F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # 将NaN转换为0
    macro_f1_score = np.mean(f1_scores)
    
    return {
        'Accuracy': accuracy,
        'Macro Precision': macro_precision,
        'Macro Recall': macro_recall,
        'Macro F1 Score': macro_f1_score
    }