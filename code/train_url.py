# coding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cluster import KMeans
from urllib.parse import unquote

import nltk
import re
import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

good_url_train = '..//data//good_fromE.txt'
bad_url_train = '..//data//badqueries.txt'
# model = '../model/svm.pickle'
model = '../model/lg.pickle'

num_clusters = 80
# ngram 系数
n = 2

#   从txt文件中获取测试数据
def get_url():

    good_url_list = []
    bad_url_list = []
    with open(good_url_train,'r') as good_url:

        for i in good_url.readlines()[:]:
            i.strip()
            good_url_list.append(i)

    with open(bad_url_train,'r') as bad_url:

        for i in bad_url.readlines()[:]:
            i.strip()
            bad_url_list.append(i)

    return [good_url_list,bad_url_list]

#   对数据进行处理
#   将所有的数据全部转化为小写
def deal_word():
    url = get_url()
    good_url_list = url[0]
    bad_url_list = url[1]
    for i in range(len(good_url_list)):
        deal_good_url = good_url_list[i].lower()
        deal_good_url = unquote(unquote(deal_good_url))
        good_url_list[i] = deal_good_url


    for i in range(len(bad_url_list)):
        deal_bad_url = bad_url_list[i].lower()
        deal_bad_url = unquote(unquote(deal_bad_url))
        bad_url_list[i] = deal_bad_url

    return [good_url_list,bad_url_list]
#  使用正则表达式进行匹配

def split_word(one_url):

    deal_word = []

    one_url=one_url.lower()

    one_url=unquote(unquote(one_url))

    one_url,num=re.subn(r'\d+',"0",one_url)

    one_url,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+',"http://u", one_url)

    r = '''(?x)[\w\.]+?\(|\)|"\w+?"|'\w+?'|http://\w|</\w+>|<\w+>|<\w+|\w+=|>|[\w\.]+'''

    word_split =  nltk.regexp_tokenize(one_url,r)

    for item in word_split:

        if item == '0' or item == 'http://u' or item == '':

            continue

        deal_word.append(item)

    return deal_word
#url_list = ['http://www.baidu.com/system/login.php?site_path=<script>alert(1)</script>','http://www.nmap.org/index.php?op=viewarticle&articleid=9999/**/union/**/select/**/1331908730,1,1,1,1,1,1,1--&blogid=1','http://www.chengdu.com/main.php?id=\'onclick=alert(1)#']
#   统计词频

# def tf_idf(word_list):
#
#     one_url_tf ={}
#
#     resetlist = list(set(word_list))
#     for i in resetlist:
#         # num = 0
#         # num = word_list.count(i)
#         one_url_tf[i] = word_list.count(i)
#
#      return one_url_tf


class Train(object):

    def __init__(self):

        self.url_list = get_url()

    def model_train(self):


        good_url_y = []
        bad_url_y = []

        for i in range(len(self.url_list[0])):

            good_url_y.append(0)

        for i in range(len(self.url_list[1])):

            bad_url_y.append(1)

        X = self.url_list[0]+self.url_list[1]
        Y = good_url_y + bad_url_y

        #   定义矢量化实例
        self.vectorizer = TfidfVectorizer(tokenizer = self.get_ngrams)
        '''
        fit_transform()
        先拟合数据，再标准化
        '''
        X = self.vectorizer.fit_transform(X)
        print(X)
        print('向量化后维度：' + str(X.shape))
        #   使用kmeans对原始的数据进行降维，以降低数据处理的复杂度
        # weight = self.kmeans(X)
        X = self.transform(self.kmeans(X))

        print('降维后的维度：' + str(X.shape))
        print(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        self.lgs = LogisticRegression(solver='liblinear')
        # self.lgs = svm.SVC()

        self.lgs.fit(X_train, y_train)

        self.lgs.score(X_test, y_test)

        print('模型的准确度:{}'.format(self.lgs.score(X_test, y_test)))

        with open(model, 'wb') as output:

            pickle.dump(self, output)


    def predict(self,new_url):

        #  加载训练好的模型
        try:
            with open(model,'rb') as f:
                self = pickle.load(f)
            print('loading model success')
        #  如果模型不存在，则进行训练
        except FileNotFoundError:
            self.model_train()

        #  对数据进行处理
        # new_url = new_url.lower()
        for i in range(len(new_url)):
            new_url[i] = new_url[i].lower()
            new_url[i] = unquote(new_url[i])

        # new_url =unquote(new_url)

        url_predict =self.vectorizer.transform(new_url)

        url_predict = self.transform(url_predict.tolil().transpose())

        res = self.lgs.predict(url_predict)

        if res == 0:
            result = 'url为正常请求'
            print("url为正常请求")
        else:
            result = 'url为恶意攻击'
            print("恶意url")

        return  result



    def get_ngrams(self,query):

        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery) - n):
            ngrams.append(tempQuery[i:i + n])

        return ngrams

    #  使用kmeas进行降维
    #  同时将其进行转化为链式的系数矩阵

    def kmeans(self,weight):

        # print('kmeans之前矩阵大小： ' + str(weight.shape))
        #  矩阵的转置
        weight = weight.tolil().transpose()
        # 同一组数据 同一个k值的聚类结果是一样的。保存结果避免重复运算
        try:

            with open('../model/train.label', 'r') as input:

                print('loading kmeans success')
                a = input.read().split(' ')

                self.label = [int(i) for i in a[:-1]]
                # for i in a[:-1]:
                #
                #     self.label = int(i)

        except FileNotFoundError:

            print('Start Kmeans ')

            '''
            n_clusters: 指定K的值
            max_iter: 对于单次初始值计算的最大迭代次数
            n_init: 重新选择初始值的次数
            init: 制定初始值选择的算法
            n_jobs: 进程个数，为-1的时候是指默认跑满CPU
            '''
            clf = KMeans(n_clusters=num_clusters, precompute_distances=False,max_iter=300, n_init=40,init='k-means++')

            s = clf.fit(weight)
            print(s)

            # 保存聚类的结果
            self.label = clf.labels_
            print(self.label)
            with open('../model/train.label', 'w') as output:
                for i in self.label:
                    output.write(str(i) + ' ')
        print('kmeans 完成,聚成 ' + str(num_clusters) + '类')
        return weight

    def transform(self, weight):

        from scipy.sparse import coo_matrix

        a = set()
        # 用coo存 可以存储重复位置的元素
        row = []
        col = []
        data = []
        # i代表旧矩阵行号 label[i]代表新矩阵的行号
        for i in range(len(self.label)):
            if self.label[i] in a:
                continue
            a.add(self.label[i])
            for j in range(i, len(self.label)):
                if self.label[j] == self.label[i]:
                    temp = weight[j].rows[0]
                    col += temp
                    temp = [self.label[i] for t in range(len(temp))]
                    row += temp
                    data += weight[j].data[0]

        newWeight = coo_matrix((data, (row, col)), shape=(num_clusters, weight.shape[1]))
        return newWeight.transpose()

# if __name__ =='__main__':
#
#     url_list = ['www.guoxiaowen.com']
#     # with open('../data/test.txt','r') as good_url:
#     #
#     #     for i in good_url.readlines()[:]:
#     #         i.strip()
#     #         url_list.append(i)
#     a = Train()
#     new_url = []
#     for i in url_list:
#         new_url.append(i)
#         print(i)
#         a.predict(new_url)
#         new_url = []


