# -*- coding: utf-8 -*-
# @Project ：AiWaf
# @Time    : 2022/5/24 23:04
# @Author  : honywen
# @FileName: loaddata.py
# @Software: PyCharm

import csv
from collections import Counter


def label2onehot(label):
    onehot_label = [0,0,0]
    onehot_label[label] = 1
    return onehot_label


#  加载XSS和SQLI的数据集
def loaddata_xss():
    data_list = list()
    lable_list = list()
    filename = "../data/XSS_dataset.csv"
    with open(filename, encoding="utf8") as f:
        csv_reader = csv.reader(f)
        for line_no, line in enumerate(csv_reader, 1):
            if line_no == 1:
                pass   # 文件头不读取
            else:
                data_list.append(line[1])                   #   0： 正常样本
                onehothlabel = label2onehot(int(line[2]))   #   1: xss恶意样本
                lable_list.append(onehothlabel)
    # result = Counter(lable_list)
    # print(result)
    return data_list,lable_list


def loaddata_sqli():
    data_list = list()
    lable_list = list()
    filename = "../data/SQLiV3.csv"
    with open(filename, encoding="utf-8", errors='ignore') as f:
        csv_reader = csv.reader(f)
        for line_no, line in enumerate(csv_reader, 1):
            if line_no == 1:
                pass   # 文件头不读取
            else:
                data = line[0]
                label = line[1]
                if data.strip() == "" or label not in ["1","0"]:
                    continue
                data_list.append(data)
                if label == 0:
                    onehothlabel = label2onehot(int(label))     #   0： 正常样本
                else:
                    onehothlabel = label2onehot(int(label)+1)    #   2: sqli恶意样本
                lable_list.append(onehothlabel)
    return data_list,lable_list
