# -*- coding: utf-8 -*-
# @Project ：AiWaf
# @Time    : 2022/5/25 19:58
# @Author  : honywen
# @FileName: vecmodel.py
# @Software: PyCharm


from gensim.corpora.dictionary import Dictionary
from keras_preprocessing import sequence
from gensim.models.word2vec import Word2Vec
from staticfeature import GeneSeg
from loaddata import loaddata_sqli, loaddata_xss


#   训练word2vec模型
def trainword2vec():
    embedding_size = 128
    skip_window = 5
    num_sampled = 64
    num_iter = 100
    data_list_sqli, lable_list_sqli = loaddata_sqli()
    data_list_xss, lable_list_xss = loaddata_xss()
    data_set = list()
    for i in data_list_sqli + data_list_xss:
        data_set.append(GeneSeg(i))
    model = Word2Vec(data_set, vector_size=embedding_size, window=skip_window, negative=num_sampled, epochs=num_iter)
    model.save('../model/model_word2vec')


#   将用户输入的单个payload转换成向量
def payload2vec(sentence):
    maxlen = 200
    gensim_dict = Dictionary()
    vecmodel = Word2Vec.load('../model/model_word2vec')
    gensim_dict.doc2bow(vecmodel.wv.key_to_index.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}
    data = list()
    sentence = GeneSeg(sentence)
    for word in sentence:
        try:
            data.append(w2indx[word])
        except:
            data.append(0)
    vec_attributes = sequence.pad_sequences([data], maxlen=maxlen,  padding='post')
    return vec_attributes


#  将data转换成vector
def data2vec(data_set):
    maxlen = 200
    data_set_vec = list()
    gensim_dict = Dictionary()
    vecmodel = Word2Vec.load('../model/model_word2vec')
    gensim_dict.doc2bow(vecmodel.wv.key_to_index.keys(), allow_update=True)
    # the index of a word which have word vector is not 0
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}
    # integrate all the corresponding word vectors into the word vector matrix
    w2vec = {word: vecmodel.wv[word] for word in w2indx.keys()}
    for payload in data_set:
        payload_vec = list()
        payload_seg = GeneSeg(payload)
        for seg in payload_seg:
            try:
                payload_vec.append(w2indx[seg])
            except:
                payload_vec.append(0)
        vec_attributes = sequence.pad_sequences([payload_vec], maxlen=maxlen, padding='post')
        data_set_vec.append(vec_attributes[0])
    return data_set_vec

# data2vec(["http://honywen.com/?AND 1 = utl_inaddr.get_host_address,<script>  ? *,$,@,! </script> ,< - + # (  (  SELECT DISTINCT "])



