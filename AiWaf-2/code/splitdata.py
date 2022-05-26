# -*- coding: utf-8 -*-
# @Project ：AiWaf
# @Time    : 2022/5/25 19:55
# @Author  : honywen
# @FileName: splitdata.py
# @Software: PyCharm


import random
from vecmodel import data2vec
from loaddata import loaddata_sqli, loaddata_xss


#   训练集测试集数据划分
def splitdata(dataloader):
    #  划分训练验证测试集  划分比例6：2：2
    len_dataloader = len(dataloader)
    trainvalidaset_num = int(len_dataloader * 0.8)
    trainvalidaset_idxs = random.sample(range(0, len_dataloader), trainvalidaset_num)

    validation_num = int(trainvalidaset_num * 0.25)
    validation_idxs = random.sample(trainvalidaset_idxs, validation_num)

    trainset_idxs = list(set(trainvalidaset_idxs) - set(validation_idxs))
    train_dataloader = [dataloader[i] for i in trainset_idxs]
    validation_dataloader = [dataloader[i] for i in validation_idxs]
    test_dataloader = list()
    for i in range(len_dataloader):
        if i in trainvalidaset_idxs:
            pass
        else:
            test_dataloader.append(dataloader[i])
    print(len(train_dataloader), len(validation_dataloader), len(test_dataloader))
    return train_dataloader, validation_dataloader, test_dataloader


def splitmain():
    data_list_sqli, lable_list_sqli = loaddata_sqli()
    data_list_xss, lable_list_xss = loaddata_xss()
    data_list_sqli_vec = data2vec(data_list_sqli)
    data_list_xss_vec = data2vec(data_list_xss)
    sqli_dataloader = list(zip(data_list_sqli_vec,lable_list_sqli))
    xss_dataloader = list(zip(data_list_xss_vec, lable_list_xss))
    sqli_train_dataloader,sqli_validation_dataloader, sqli_test_dataloader = splitdata(sqli_dataloader)
    xss_train_dataloader, xss_validation_dataloader, xss_test_dataloader = splitdata(xss_dataloader)
    train_dataloader = sqli_train_dataloader + xss_train_dataloader
    validation_dataloader = sqli_validation_dataloader + xss_validation_dataloader
    test_dataloader = sqli_test_dataloader + xss_test_dataloader
    #   打乱数据顺序
    random.shuffle(train_dataloader)
    random.shuffle(validation_dataloader)
    random.shuffle(test_dataloader)
    train_data = [i[0] for i in train_dataloader]
    train_label = [i[1] for i in train_dataloader]
    validation_data = [i[0] for i in validation_dataloader]
    validation_label = [i[1] for i in validation_dataloader]
    test_data = [i[0] for i in test_dataloader]
    test_label = [i[1] for i in test_dataloader]
    return train_data,train_label,validation_data,validation_label,test_data,test_label


