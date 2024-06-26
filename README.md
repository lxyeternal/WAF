# 项目概述

本项目实现了两种基于机器学习的 Web 应用防火墙（WAF），专注于检测并防御 XSS 和 SQL 注入攻击。

## AiWaf-1: 基于聚类的检测

AiWaf-1 利用聚类算法来识别和防御 XSS 及 SQL 注入攻击，提供基础防护措施。

## AiWaf-2: 基于多模型机器学习的高级检测

AiWaf-2 是一个高级的机器学习系统，使用多种模型来检测三种类别的网络行为：XSS攻击、SQL注入攻击和良性请求。该系统旨在提供高精度的威胁识别和分类，包括：

- **GRU (门控循环单元)**
- **CNN (卷积神经网络)**
- **KNN (K-最近邻)**
- **SVM (支持向量机)**
- **RF (随机森林)**

### 检测流程

1. **数据加载**：加载预定义的数据集，包含 XSS、SQL 注入以及良性样本。
2. **数据预处理**：包括 URL 解码和转换为小写处理。
3. **向量化**：使用预训练的 Word2Vec 模型进行向量化，并进行 padding 补齐。
4. **模型训练**：在三类数据上训练模型。
5. **模型预测**：利用训练好的模型进行预测。
6. **模型评估**：评估模型性能，包括生成混淆矩阵图，以可视化不同类别的识别效果。

### 快速开始

AiWaf-2 项目配置了专用的 Conda 环境，以确保所有依赖都得到正确管理。

#### 设置 Conda 环境

启动环境并运行 AiWaf-2:

```bash
conda env create -f environment.yml
conda activate aiwaf-2-env
```


### 运行项目

使用以下命令从 trainmain.py，项目的主入口文件，开始运行 AiWaf-2:

```python
python trainmain.py
```