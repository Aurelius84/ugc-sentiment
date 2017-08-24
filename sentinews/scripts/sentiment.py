# -*- coding:utf-8 -*-

import os
import pickle
import sys

import jieba
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.assemble import assemble
from utils.dataHelper import load_var, pad_sentences


class sentiment(object):
    def __init__(self,
                 model_flag='deep_lstm',
                 model_dir='../docs/model/checkpoints'):
        self.params = pickle.load(open(model_dir + '/params.pkl', 'rb'))
        self.vocabulary_inv = load_var(model_dir + '/vocabulary_inv.txt')
        self.vocabulary = {wd: i for i, wd in enumerate(self.vocabulary_inv)}
        self.labels = ["消极", "中性", "积极"]
        # assemble model
        self.model = assemble(model_flag, self.params)
        # load trained weights
        self.model.load_weights(model_dir + '/default_best.h5')

    def predict(self, datas):
        """
        1. process data
        2. predict probs
        3. return json data
        """
        process_data = self.process(datas)
        probs = self.model.predict(np.array(process_data))
        probs = probs * (probs > 0.)
        probs[:, 1] += 0.6
        probs[:, 2] -= 0.3
        preds = np.argmax(probs, axis=1).flatten()
        # return result
        result = []
        for i in range(len(probs)):
            json_data = {
                "labels": self.labels[preds[i]],
                "probs": {self.labels[j]: probs[i][j]
                          for j in range(3)}
            }
            result.append(json_data)
        # single data return json
        if len(datas) == 1:
            return result[0]
        # multi data return list of json
        return result

    def process(self, datas):
        """
        1. segment using jieba
        2. transform word to word_index
        3. padding_word
        """
        if not isinstance(datas, list):
            datas = [datas]
        self.seg_data = [jieba.lcut(data) for data in datas]
        data = [[self.vocabulary[wd] for wd in sen if wd in self.vocabulary] for sen in self.seg_data]
        pad_data = pad_sentences(
            data,
            padding_word=0,
            mode=self.params['X']['sequence_length'])

        return pad_data

    @property
    def words(self):
        return self.seg_data


if __name__ == '__main__':
    from sklearn.metrics import classification_report
    senti = sentiment()
    data = u"最近很破坏，很生气，怎么可以这么缺德。"
    test_datas = open('../docs/sentiment/keySentence.txt', 'r').readlines()
    cate = {'-1': u'消极', '0': u'中性', '1': u'积极'}
    conts = [x.decode('utf8').split('|')[1] for x in test_datas]
    labels = [cate[x.split('|')[0]] for x in test_datas]
    result = senti.predict(test_datas)
    pre = [res['labels'].decode('utf-8') for res in result]
    for i in range(30):
        print(conts[i])
        print('target: %s' % labels[i])
        print('predict: %s' % pre[i])
        s = ['%s: %s' % (result[i]['probs'].keys()[j], result[i]['probs'].values()[j]) for j in range(3)]
        print(' '.join(s))
    print(classification_report(labels, pre))
