# coding: utf-8
import os
import sys
from snownlp import SnowNLP
from sklearn.metrics import classification_report
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.dataHelper import corpus_balance


datas = open('../docs/keySentence.txt', 'r').readlines()
labels = [data.decode('utf-8').split('|')[0] for data in datas]
contents = [data.decode('utf-8').strip('\n').split('|')[1] for data in datas]

contents, labels = corpus_balance(contents, labels, mod='average')
print(labels[0])
print(contents[0])
s = SnowNLP('天气如何呢？')
print(s.sentiments)
