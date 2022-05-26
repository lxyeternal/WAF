本项目实现了两种基于机器学习的WAF

## AiWaf-1

基于聚类的XSS和SQL注入检测

## AiWaf-2

基于机器学习的XSS和SQL注入检测

实现了基于GRU，CNN，KNN，SVM，RF共五个检测模型

检测过程：数据加载-》数据预处理(urldecode和转小写)->向量化（预训练word2Vec模型，padding补齐）->模型训练->模型预测->模型评估