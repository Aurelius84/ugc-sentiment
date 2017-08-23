# coding: utf-8
import os
import sys
from collections import Counter
import numpy as np
import jieba

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.CNN_RNN import Text_CNN_RNN
from utils.dataHelper import load_data_and_labels, segDataset, corpus_balance


def train(params, category=None, model_dir='../docs/model/'):

    # 加载数据
    datas, labels = load_data_and_labels('../docs/sentiment/keySentence.txt', split_tag='|')
    jieba.load_userdict('../docs/sentiment/kw_dict.txt')
    datas = [list(jieba.cut(x[0])) for x in datas]
    labels = [y[0] for y in labels]
    # 转化为0级数值标签
    print(Counter(labels))
    # exit()
    # one vs rest corpus balanced
    trn_datas, trn_labels = corpus_balance(datas, labels, mod='average')
    print(Counter(trn_labels))
    # 文档分词
    print(trn_datas[:4])
    print(trn_labels[:4])
    # CNN
    cnn_model = Text_CNN_RNN(params)
    cnn_model.train(trn_datas, trn_labels, model_dir, 'debug')


if __name__ == '__main__':
    # CNN params
    params = {
        "batch_size": 16,
        "dropout_keep_prob": 0.7,
        "embedding_dim": 8,
        "evaluate_every": 5,
        "filter_sizes": "3,4,5",
        "hidden_unit": 128,
        "l2_reg_lambda": 0.1,
        "max_pool_size": 3,
        "non_static": True,
        "num_epochs": 30,
        "num_filters": 128,
        "padding_mod": 100,
        "early_stop_acc": 0.94,
        "early_stop_step": 380
    }
    train(params)
