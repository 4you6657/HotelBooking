import nltk
from nltk.corpus import brown
import jieba
from jieba.analyse import *
import matplotlib.image as mpimg
import stylecloud
import csv
import re

#数据载入
#提取文本信息
def readFile(path):
    str_doc = ""
    with open(path,'r',encoding='utf-8') as f:
        str_doc = f.read()
        return str_doc
    #读取文本
    path = 'D:/data/txt/nahan.txt'
    str_doc = readFile(path)
    print(str_doc)

def textParse(str_doc):
#通过正则表达式过滤掉特殊符号、标点、英文、数字等.
# r1='[a-zA-Z0-9’!”#$%\&()*+,-./:：;；|<=>?@，-。、……【】《》？‘’“”]'
#r1 = "\W"
#去除换行符
#str_doc = re.sub(r1,' ', str_doc)
#多个空格成1个
#str_doc = re.sub(r2,' ', str_doc)
#return str_doc

#正则清洗字符串
word_list=textParse(str_doc)
print(word_list)

#中文分词jieba
#import jieba
#print(jieba.cut(str_doc))