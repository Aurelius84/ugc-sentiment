# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: PyCharm Community Edition
@file: adios_train.py
@time: 17/05/03 17:39
"""

import os
import sys

import yaml
import numpy as np
import jieba
from collections import Counter
from sklearn import preprocessing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.dataHelper import load_data_and_labels, save_var

reload(sys)
sys.setdefaultencoding('utf8')


def loadSentimentVector(file_name):
    """
    Load sentiment vector
    [Surprise, Sorrow, Love, Joy, Hate, Expect, Anxiety, Anger]
    """
    contents = [
        line.strip('\n').split() for line in open(file_name, 'r').readlines()
    ]
    sentiment_dict = {
        line[0].decode('utf-8'): [float(w) for w in line[1:]]
        for line in contents
    }
    return sentiment_dict


if __name__ == '__main__':
    # load sentiment vector
    sentiment_dict = loadSentimentVector('../docs/sentiment/extend_dict.txt')
    # use sentiment_dict as vocabulary
    vocabulary_inv = ["<PAD/>"] + list(sentiment_dict.keys())
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    # Load the dataset
    texts, labels = load_data_and_labels('../docs/sentiment/keySentence.txt', split_tag='|')
    # add user dict
    map(jieba.add_word, sentiment_dict.keys())
    texts = [jieba.lcut(x[0]) for x in texts]
    labels = [y[0] for y in labels]
    lb_wd = ['Surprise', 'Sorrow', 'Love', 'Joy', 'Hate', 'Expect', 'Anxiety', 'Anger']
    average = []
    ind = 5
    for wd in texts[ind]:
        if wd in sentiment_dict:
            s = ['%s:%.3f' % (lb_wd[i], sentiment_dict[wd][i]) for i in range(6)]
            average.append(sentiment_dict[wd])
            print(wd)
            print(s)
    average = np.mean(np.array(average), axis=1)
    s = ['%s:%.3f' % (lb_wd[i], average[i]) for i in range(6)]
    print('average:')
    print(s)
    print(labels[ind])
    print(' '.join(texts[ind]))

    exit()

    labels = [x[0] for x in labels]
    # filter by sentiment_dict
    texts = [filter(lambda x: x != 0, content) for content in texts]
    print(texts[:3])
    # padding
    texts = pad_sentences(texts, padding_word=0, mode='average')
    texts, labels = corpus_balance(texts, labels, mod='average')
    # save vocab
    save_var('../docs/model/checkpoints/vocabulary_inv', vocabulary_inv)
    category = ['-1', '0', '1']
    # [0] -> 0
    print(Counter(labels))
    # save category
    save_var('../docs/model/checkpoints/category', category)
    # vectorize
    labelEncoder = preprocessing.label_binarize
    _labels = labelEncoder(labels, classes=category)
    _texts = np.array(texts)
    # print(_labels[100:300])
    # exit()

    ind = np.arange(_texts.shape[0])
    np.random.shuffle(ind)
    ratio = 0.3
    split_n = int(ratio * len(_texts))
    trn_labels = _labels[ind[split_n:]]
    trn_text = _texts[ind[split_n:]]
    tst_labels = _labels[ind[:split_n]]
    tst_text = _texts[ind[:split_n]]

    print('train data size : %d , test data size : %d' % (len(trn_labels),
                                                          len(tst_labels)))
    print('X sequence_length is : %d , Y dim : %d' % (trn_text.shape[1],
                                                      trn_labels.shape[1]))
    # load params config
    params = yaml.load(open('../utils/config.yaml'))
    params['X']['sequence_length'] = trn_text.shape[1]
    params['X']['vocab_size'] = len(vocabulary)
    params['Y']['dim'] = len(category)

    print(trn_text[0])
    print(trn_labels[0])
    ratio = 0.5
    valid_N = int(ratio * tst_text.shape[0])
    train_dataset = {'X': trn_text, 'Y': trn_labels}
    valid_dataset = {'X': tst_text[:valid_N], 'Y': tst_labels[:valid_N, :]}
    test_dataset = {'X': tst_text[valid_N:], 'Y': tst_labels[valid_N:, :]}
    # start train
    train(train_dataset, valid_dataset, test_dataset, params, model_flag='deep_lstm')
