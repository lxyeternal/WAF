# -*- coding: UTF-8 -*-

from train_url import *

xss_high = ['<scrip', '</script', '<iframe' '</iframe','response','write(','eval(','prompt(','alert(','javascript;','document','cookie']
xss_middle = ['Onclick=','onerror=','<!–','–>','<base','</base>>','location','hash','window','name','<form','</form']
xss_low = ['echo','print','href=','sleep']


sql_high = ['and','or','xp_','substr','utl_','benchmark','shutdown','@@version','information_schema','hex(']
sql_middle = ['select','if(','union','group','by','–','count(','/**/','char(','drop','delete','concat','orderby','case' 'when','assic(','exec(','length']
sql_low = ['and','or','like','from','insert', 'update','create','else', 'exist','table' ,'database','where','sleep','mid','updatexml(','null','sqlmap','md5(','floorm','rand','cast','dual','fetch','print','declare','cursor','extractvalue(','upperjoin','exec','innier','convert','distinct']

def find_type(url_list_char):

    attack_rank = ''

    for i in url_list_char:

        if i in xss_high:
            attack_rank = '攻击类型：XSS  攻击等级：严重'
            break
        elif i in xss_middle:
            attack_rank = '攻击类型：XSS  攻击等级：中级'
            break
        elif i in xss_low:
            attack_rank = '攻击类型：XSS  攻击等级：轻微'
            break
        elif i in sql_high:
            attack_rank = '攻击类型：SQL注入  攻击等级：严重'
            break
        elif i in sql_middle:
            attack_rank = '攻击类型：SQL注入  攻击等级：中级'
            break
        elif i in sql_low:
            attack_rank = '攻击类型：SQL注入  攻击等级：轻微'
            break
        else:
            attack_rank = '攻击类型：未知  攻击等级：未知'

    return attack_rank
