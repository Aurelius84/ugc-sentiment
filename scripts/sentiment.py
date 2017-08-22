# -*- coding:utf-8 -*-

import os
import pickle
import sys

import jieba
import numpy as np
from utils.assemble import assemble
from utils.dataHelper import load_var, pad_sentences

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class sentiment(object):
    def __init__(self,
                 model_flag='deep_lstm',
                 model_dir='../docs/model/checkpoints'):
        self.params = pickle.load(open(model_dir + '/params', 'rb'))
        self.vocabulary_inv = load_var(model_dir + '/vocabulary_inv')
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
        probs = self.model.predict(process_data)
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
        2. padding_word
        3. transform word to word_index
        """
        self.seg_data = [jieba.lcut(data) for data in datas]
        pad_data = pad_sentences(
            self.seg_data,
            padding_word="<PAD/>",
            mode=self.params['sequence_length'])
        pad_data = [[self.vocabulary[wd] for wd in sen] for sen in pad_data]

        return pad_data

    @property
    def words(self):
        return self.seg_data
