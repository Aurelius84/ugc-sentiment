# -*- coding:utf-8 -*-

import os
import pickle

from . import utils

import jieba
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs/model')


class SentiNews(object):
    def __init__(self,
                 model_flag='deep_lstm',
                 model_dir=MODEL_DIR):
        self.params = pickle.load(open(model_dir + '/params', 'rb'))
        self.vocabulary_inv = utils.dataHelper.load_var(
            model_dir + '/vocabulary_inv')
        self.vocabulary = {wd: i for i, wd in enumerate(self.vocabulary_inv)}
        self.labels = [u"消极", u"中性", u"积极"]
        # assemble model
        self.model = utils.assemble.assemble(model_flag, self.params)
        # load trained weights
        self.model.load_weights(model_dir + '/default_best.h5')

    def sentiment(self, datas):
        """
        1. process data
        2. predict probs
        3. return json data
        """
        process_data = self.process(datas)
        probs = self.model.predict(np.array(process_data))
        # probs = probs * (probs > 0.)
        # probs[:, 1] += 0.6
        # probs[:, 2] -= 0.3
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
        if not isinstance(datas, list):
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
        data = [[self.vocabulary[wd] for wd in sen if wd in self.vocabulary]
                for sen in self.seg_data]
        pad_data = utils.dataHelper.pad_sentences(
            data, padding_word=0, mode=self.params['X']['sequence_length'])

        return pad_data

    @property
    def words(self):
        return self.seg_data
