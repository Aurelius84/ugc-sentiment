# -*- coding:utf-8 -*-

import os
import sys

import yaml
import numpy as np
import json
import pickle
from copy import deepcopy
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import classification_report

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad, RMSprop
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.assemble import assemble
from utils.dataHelper import load_data, save_var, corpus_balance, pad_sentences

# Assemble and compile the model
model = assemble(model_flag, params)

# complie model
model.compile(
    loss=params['Y']['loss_func'],
    metrics=['accuracy'],
    optimizer=optimizer(params['iter']['learn_rate']))

# Test the model
probs = model.predict(
    test_dataset, verbose=1, batch_size=params['iter']['batch_size'])
preds = np.argmax(probs, axis=1).flatten()
targets = np.argmax(test_dataset['Y'], axis=1).flatten()

print(classification_report(targets, preds, target_names=category))

if __name__ == '__main__':
    # load sentiment vector
    sentiment_dict = loadSentimentVector('../docs/kw_dict.txt')
    # use sentiment_dict as vocabulary
    vocabulary_inv = ["<PAD/>"] + list(sentiment_dict.keys())
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
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
