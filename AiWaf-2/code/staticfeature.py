# -*- coding: utf-8 -*-
# @Project ：AiWaf
# @Time    : 2022/5/25 18:26
# @Author  : honywen
# @FileName: staticfeature.py
# @Software: PyCharm

import nltk
import re
from urllib.parse import unquote


def get_last_char(url):
    if re.search('/$', url, re.IGNORECASE):
        return 1
    else:
        return 0

def get_evil_word(url):
    # /* 是注释
    # 空格不要用
    # |(src=)
    return len(re.findall("(/\*)|(alert)|(script)|(onerror)|(onload)|(eval)|(prompt)|(document)|(window)|(confirm)|(onmouseover)|(onclick)|(console)|(onfocus)|(setinterval)|(settimeout)", url, re.IGNORECASE))


def countfeature(payload):

    payload = unquote(payload)
    num_len = len(re.compile(r'\d').findall(payload))
    if len(payload) != 0:
        num_f = num_len / len(payload)  # 数字字符频率
    capital_len = len(re.compile(r'[A-Z]').findall(payload))
    if len(payload) != 0:
        capital_f = capital_len / len(payload)  # 大写字母频率
    line = payload.lower()

    #  sql注入的静态特征
    key_num = line.count('and') + line.count('or') + line.count('xor') + line.count(
        'sysobjects') + line.count('version') + line.count('substr') + line.count('len') + line.count(
        'substring') + line.count('exists')
    key_num = key_num + line.count('mid') + line.count('asc') + line.count('inner join') + line.count(
        'xp_cmdshell') + line.count('version ') + line.count('exec') + line.count('having ') + line.count(
        'unnion') + line.count('order') + line.count('information schema')
    key_num = key_num + line.count('load_file') + line.count('load data infile') + line.count(
        'into outfile') + line.count('into dumpfile')
    if len(line) != 0:
        space_f = (line.count(" ") + line.count(" ")) / len(line)  # 空格百分比
        special_f = (line.count("{") * 2 + line.count('28%') * 2 + line.count('NULL') + line.count('[') + line.count(
            '=') + line.count('?')) / len(line)
        symbol_f = (line.count("'") + line.count("-") + line.count("+") + line.count("#") + line.count(",") + line.count("@") + line.count("*") + line.count(">") + line.count("<") + line.count("||") + line.count("\\") + line.count("/")) / len(line)
        prefix_f = (line.count('\\x') + line.count('&') + line.count('\\u') + line.count('%')) / len(line)

    #  xss的静态特征
    xss_key_num = line.count('java') + line.count('script') + line.count('iframe') + line.count('body') + line.count('style') + \
        line.count('and ') +line.count('and ') + line.count('and ')

    print('%f,%f,%f,%f,%f,%f,%f' % (key_num,symbol_f,capital_f,num_f,space_f,special_f,prefix_f))

def GeneSeg(payload):
    #数字泛化为"0"
    payload=payload.lower()
    payload=unquote(unquote(payload))
    payload,num=re.subn(r'\d+',"0",payload)
    #替换url为”http://u
    payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\#,-,+,\<,\>,\',\",\\,\*,$,@,!,\?,\ ]
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)

# print(GeneSeg(" AND 1 = utl_inaddr.get_host_address,<script>  ? *,$,@,! </script> ,< - + # (  (  SELECT DISTINCT ( PASSWORD )  FROM  ( SELECT DISTINCT ( PASSWORD ) , ROWNUM AS LIMIT FROM SYS.USER$ )  WHERE LIMIT = 8  )  )   AND 'i' = 'i,1"))
