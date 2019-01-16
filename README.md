# WAF

环境配置：

首先要求：python 3.6
响应的库函数：TensorFlow，python-scapy


train_url.py：
	该文件主要包含的功能是实现对训练数据的加载以及处理，同时训练模型模块以及预测分析模块也在此
get_url.py:
	该文件的主要功能是实现抓取数据包，同时将数据包中的URL解析出来
type.py：
	该文件的主要功能是实现对攻击类型的判断
UI.py：
	该文件是实现UI界面
Main.py：
	主函数的执行入口
