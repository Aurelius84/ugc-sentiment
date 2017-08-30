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

reload(sys)
sys.setdefaultencoding('utf8')


def train(train_dataset,
          valid_dataset,
          test_dataset,
          params,
          model_flag='deep_conv'):

    # Assemble and compile the model
    model = assemble(model_flag, params)
    raw_test_dataset = deepcopy(test_dataset)
    # Prepare embedding layer weights and convert inputs for static model
    model_type = params['iter']['model_type']
    print("Model type is", model_type)
    if model_type == "CNN-non-static" or model_type == "CNN-static":
        # embedding_weights = train_word2vec(
        #     np.vstack((valid_dataset['X'], test_dataset['X'])),
        #     vocabulary_inv,
        #     num_features=params['X']['embedding_dim'],
        #     min_word_count=1,
        #     context=5)
        # embedding_weights = [np.ones((len(vocabulary_inv), params['X']['embedding_dim']))]
        # for i, w in enumerate(vocabulary_inv):
        #     if w in sentiment_dict:
        #         embedding_weights[0][i] = np.dot(np.array(sentiment_dict[w]), embedding_weights[0][i])
        #     else:
        #         embedding_weights[0][i] = np.dot(np.random.uniform(0, 0.01, 6), embedding_weights[0][i])
        embedding_weights = [
            np.array([
                np.array(sentiment_dict[w]) if w in sentiment_dict else
                np.random.uniform(0, 0., params['X']['embedding_dim']) for w in vocabulary_inv
            ])
        ]

        if model_type == "CNN-static":
            train_dataset['X'] = embedding_weights[0][train_dataset['X']]
            test_dataset['X'] = embedding_weights[0][test_dataset['X']]
            valid_dataset['X'] = embedding_weights[0][valid_dataset['X']]
        elif params['iter']['model_type'] == "CNN-non-static":
            embedding_layer = model.get_layer('embedding')
            embedding_layer.set_weights(embedding_weights)
    elif model_type == "CNN-rand":
        embedding_weights = None
    else:
        raise ValueError("Unknown model type")
    # complie model
    optimizer = RMSprop if 'lstm' in model_flag else Adagrad
    pickle.dump(params, open('../docs/model/params', 'wb'))
    model.compile(
        loss=params['Y']['loss_func'],
        metrics=['accuracy'],
        optimizer=optimizer(params['iter']['learn_rate']))

    # Make sure checkpoints folder exists
    model_dir = params['iter']['model_dir']
    model_name = params['iter']['model_name']
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            model_dir + model_name,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='min'),
        EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='min'),
    ]

    # Fit the model to the data
    batch_size = params['iter']['batch_size']
    nb_epoch = params['iter']['epoch']

    # start to train
    model.fit(
        x=train_dataset["X"],
        y=train_dataset['Y'],
        validation_data=(valid_dataset["X"], valid_dataset["Y"]),
        batch_size=batch_size,
        epochs=nb_epoch,
        callbacks=callbacks,
        verbose=1)

    # Load the best weights
    if os.path.isfile(model_dir + model_name):
        model.load_weights(model_dir + model_name)
    # Test the model
    probs = model.predict(
        test_dataset, verbose=1, batch_size=params['iter']['batch_size'])
    preds = np.argmax(probs, axis=1).flatten()
    targets = np.argmax(test_dataset['Y'], axis=1).flatten()
    for i in range(60):
        print('\n')
        print(' '.join([
            vocabulary_inv[ii]
            for ii in filter(lambda ind: ind != 0, raw_test_dataset['X'][i])
        ]))
        print('target: %s' % category[targets[i]])
        print('predict: %s' % category[preds[i]])
        print('probs: %s' % probs[i])

    print(classification_report(targets, preds, target_names=category))


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

    # Load the datasets
    texts, labels, vocabulary, vocabulary_inv = load_data(
        '../docs/sentiment/cec.txt',
        use_tst=False,
        lbl_text_index=[0, 1],
        split_tag='|',
        padding_mod='max',
        use_jieba_segment=True,
        vocabulary=vocabulary,
        vocabulary_inv=vocabulary_inv
        )

    labels = [x[0] for x in labels]
    # filter by sentiment_dict
    texts = [filter(lambda x: x != 0, content) for content in texts]

    # padding
    texts = pad_sentences(texts, padding_word=0, mode=100)
    print(Counter(labels))
    texts, labels = corpus_balance(texts, labels, mod='average')
    # save vocab
    save_var('../docs/model/vocabulary_inv', vocabulary_inv)
    category = ['-1', '0', '1']
    # [0] -> 0
    print(Counter(labels))
    # save category
    save_var('../docs/model/category', category)
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
