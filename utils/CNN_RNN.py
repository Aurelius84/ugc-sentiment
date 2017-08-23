# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: PyCharm Community Edition
@file: CNN_RNN.py
@time: 16/12/18 下午7:42
"""
import os
import sys

import json
import shutil
import pickle
import logging
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from dataHelper import segDataset, cnn_data_helper, readExcelByCol, clean_str

logging.getLogger().setLevel(logging.INFO)

reload(sys)
sys.setdefaultencoding("utf-8")


class CNNRNN(object):
    """
    CNNRNN  ———— 卷积神经网络 抽象类
            ———— 相对普通的词权重模型,考虑了上下文信息
    """

    def __init__(self,
                 embedding_mat,
                 non_static,
                 hidden_unit,
                 sequence_length,
                 max_pool_size,
                 num_classes,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda=0.0):
        '''
        初始化即构建模型

        :param embedding_mat: 词典词向量矩阵
        :param non_static: 是否为静态
        :param hidden_unit: 隐层单元数
        :param sequence_length: 文档集长度
        :param max_pool_size: 最大池化尺度
        :param num_classes: 分类类别个数
        :param embedding_size: 词向量维度
        :param filter_sizes: 卷积模板尺度
        :param num_filters: 卷积模板个数
        :param l2_reg_lambda:
        :return:
        '''

        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [])
        self.pad = tf.placeholder(
            tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if not non_static:
                W = tf.constant(embedding_mat, name='W')
            else:
                W = tf.Variable(embedding_mat, name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            emb = tf.expand_dims(self.embedded_chars, -1)

        pooled_concat = []
        # reduced = np.int32(np.ceil((sequence_length) * 1.0 / max_pool_size))
        reduced = np.int32(
            np.ceil((sequence_length - max_pool_size + 1) / 2.) + 1)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):

                # Zero paddings so that the convolution output have dimension
                # batch x sequence_length x emb_size x channel
                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio
                pad_prio = tf.concat([self.pad] * num_prio, 1)
                pad_post = tf.concat([self.pad] * num_post, 1)
                emb_pad = tf.concat([pad_prio, emb, pad_post], 1)

                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(
                    tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    emb_pad,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_pool_size, 1, 1],
                    strides=[1, 2, 1, 1],
                    padding='SAME',
                    name='pool')
                pooled = tf.reshape(pooled, [-1, reduced, num_filters])
                pooled_concat.append(pooled)

        pooled_concat = tf.concat(pooled_concat, 2)
        pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob)

        inputs = [
            tf.squeeze(input_, [1])
            for input_ in tf.split(
                pooled_concat, num_or_size_splits=int(reduced), axis=1)
        ]

        # lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_unit)
        lstm_cell = tf.contrib.rnn.GRUCell(num_units=hidden_unit)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=0.8,
            state_keep_prob=1.0,
            output_keep_prob=0.7)
        # lstm_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_cell, 10)
        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        outputs, state = tf.contrib.rnn.static_rnn(
            lstm_cell,
            inputs,
            initial_state=self._initial_state,
            sequence_length=self.real_len)

        # Collect the appropriate last words into variable output (dimension =
        # batch x embedding_size)
        output = outputs[i]
        # alpha = 0.96
        with tf.variable_scope('Output'):
            tf.get_variable_scope().reuse_variables()
            one = tf.ones([1, hidden_unit], tf.float32)
            length = len(outputs)
            for i in range(1, length):
                ind = self.real_len < (i + 1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, one)
                output = tf.add(
                    tf.multiply(output, mat),
                    tf.multiply(outputs[i], 1.0 - mat))

        with tf.name_scope('output'):
            self.W = tf.Variable(
                tf.truncated_normal([hidden_unit, num_classes], stddev=0.1),
                name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.relu(
                tf.nn.xw_plus_b(output, self.W, b), name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name='accuracy')

        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))


class Text_CNN_RNN(object):
    def __init__(self, params):
        # 声明cnn数据处理类
        self.dataHelper = cnn_data_helper()
        # 参数配置
        self.params = {
            "batch_size": params.get('batch_size', 128),
            "dropout_keep_prob": params.get('dropout_keep_prob', 0.5),
            "embedding_dim": params.get('embedding_dim', 100),
            "evaluate_every": params.get('evaluate_every', 5),
            "filter_sizes": params.get('filter_sizes', "3,4,5"),
            "hidden_unit": params.get('hidden_unit', 200),
            "l2_reg_lambda": params.get('l2_reg_lambda', 0.01),
            "max_pool_size": params.get('max_pool_size', 3),
            "non_static": params.get('non_static', False),
            "num_epochs": params.get('num_epochs', 50),
            "num_filters": params.get('num_filters', 200),
            "padding_mod": params.get('padding_mod', 'average'),
            "early_stop_acc": params.get('early_stop_acc', 0.95),
            "early_stop_step": params.get('early_stop_step', 100)
        }

    def train(self, train_x_raw, train_y_raw, model_dir, save_flag):
        '''
        模型训练与保存

        :param train_x_raw:训练文档-词集
        :param train_y_raw:训练标签集
        :return:
        '''
        # 数据预处理,包括padding和向量化
        train_x, train_y = self.dataHelper.fit(
            train_x_raw, train_y_raw, padding_mod=self.params['padding_mod'])

        # Assign a 100 dimension vector to each word by word2vec
        word_embeddings = self.dataHelper.load_embeddings(
            dim=self.params['embedding_dim'])
        embedding_mat = [
            word_embeddings[word]
            for index, word in enumerate(self.dataHelper._vocabulary_inv)
        ]
        embedding_mat = np.array(embedding_mat, dtype=np.float32)

        # 划分为训练与测试
        # train : test : dev = 8 : 1 : 1
        x, x_test, y, y_test = train_test_split(
            train_x, train_y, test_size=0.1)
        x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)

        logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(
            len(x_train), len(x_dev), len(x_test)))
        logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(
            len(y_train), len(y_dev), len(y_test)))

        # 创建模型保存目录
        trained_dir = '%s/trained_results_%s/' % (model_dir, save_flag)
        if os.path.exists(trained_dir):
            shutil.rmtree(trained_dir)
        os.makedirs(trained_dir)

        # 借助tensorflow进行流程化管理
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # CNNRNN抽象类
                cnn_rnn = CNNRNN(
                    embedding_mat=embedding_mat,
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    non_static=self.params['non_static'],
                    hidden_unit=self.params['hidden_unit'],
                    max_pool_size=self.params['max_pool_size'],
                    filter_sizes=map(int,
                                     self.params['filter_sizes'].split(",")),
                    num_filters=self.params['num_filters'],
                    embedding_size=self.params['embedding_dim'],
                    l2_reg_lambda=self.params['l2_reg_lambda'])

                global_step = tf.Variable(
                    0, name='global_step', trainable=False)
                optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
                grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
                train_op = optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)
                # Checkpoint files will be saved in this directory during
                # training
                checkpoint_dir = '%s/checkpoints_%s/' % (model_dir, save_flag)
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                os.makedirs(checkpoint_dir)
                checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

                def real_len(batches):
                    return [
                        np.ceil((np.argmin(batch + [0]) -
                                 self.params['max_pool_size'] + 1) / 2. + 1)
                        for batch in batches
                    ]

                def train_step(x_batch, y_batch):
                    '''train数据集的每轮训练'''
                    feed_dict = {
                        cnn_rnn.input_x:
                        x_batch,
                        cnn_rnn.input_y:
                        y_batch,
                        cnn_rnn.dropout_keep_prob:
                        self.params['dropout_keep_prob'],
                        cnn_rnn.batch_size:
                        len(x_batch),
                        cnn_rnn.pad:
                        np.zeros(
                            [len(x_batch), 1, self.params['embedding_dim'],
                             1]),
                        cnn_rnn.real_len:
                        real_len(x_batch),
                    }
                    _, step, loss, accuracy = sess.run([
                        train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy
                    ], feed_dict)
                    return accuracy, loss

                def dev_step(x_batch, y_batch):
                    '''dev每轮训练'''
                    feed_dict = {
                        cnn_rnn.input_x:
                        x_batch,
                        cnn_rnn.input_y:
                        y_batch,
                        cnn_rnn.dropout_keep_prob:
                        1.0,
                        cnn_rnn.batch_size:
                        len(x_batch),
                        cnn_rnn.pad:
                        np.zeros(
                            [len(x_batch), 1, self.params['embedding_dim'],
                             1]),
                        cnn_rnn.real_len:
                        real_len(x_batch),
                    }
                    step, loss, accuracy, num_correct, predictions = sess.run([
                        global_step, cnn_rnn.loss, cnn_rnn.accuracy,
                        cnn_rnn.num_correct, cnn_rnn.predictions
                    ], feed_dict)
                    return accuracy, loss, num_correct, predictions

                saver = tf.train.Saver(
                    tf.global_variables(), write_version=tf.train.SaverDef.V1)
                # saver = tf.train.Saver(tf.all_variables())
                sess.run(tf.global_variables_initializer())

                # 训练正式开始
                train_batches = self.dataHelper.batch_iter(
                    list(zip(x_train, y_train)), self.params['batch_size'],
                    self.params['num_epochs'])
                # 准确率记录
                best_accuracy, best_at_step, min_f1 = 0, 0, 0

                # Train the model with x_train and y_train

                for train_batch in train_batches:
                    x_train_batch, y_train_batch = zip(*train_batch)
                    train_acc, train_loss = train_step(x_train_batch, y_train_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    # Evaluate the model with x_dev and y_dev
                    if current_step % self.params['evaluate_every'] == 0:

                        dev_batches = self.dataHelper.batch_iter(
                            list(zip(x_dev, y_dev)), self.params['batch_size'],
                            1)

                        total_dev_correct = 0
                        pred_y = []
                        y_true = []
                        dev_loss = 0.
                        for dev_i, dev_batch in enumerate(dev_batches):
                            x_dev_batch, y_dev_batch = zip(*dev_batch)
                            # 调用dev_step 函数
                            acc, loss, num_dev_correct, predictions = dev_step(
                                x_dev_batch, y_dev_batch)
                            y_batch = [np.argmax(y) for y in y_dev_batch]
                            y_true.extend(y_batch)
                            pred_y.extend(predictions)
                            dev_loss += loss
                            total_dev_correct += num_dev_correct
                        accuracy = float(total_dev_correct) / len(y_dev)
                        logging.info('train loss: {:.4f} dev loss: {:.4f} train accuracy : {:.4f}  dev accuracy : {:.4f}'.format(
                            train_loss, dev_loss / dev_i, train_acc, accuracy))
                        r_acc, r_recall, r_f1, r_n = metrics.precision_recall_fscore_support(
                            y_true, pred_y)

                        # 每次保存最好的模型
                        if accuracy >= best_accuracy and (r_f1 >=
                                                          min_f1).all():
                            best_accuracy, best_at_step = accuracy, current_step
                            path = saver.save(
                                sess,
                                checkpoint_prefix,
                                global_step=current_step)
                            logging.critical('Saved model {} at step {}'.
                                             format(path, best_at_step))
                            logging.critical(
                                'Best accuracy {} at step {}'.format(
                                    best_accuracy, best_at_step))
                            min_f1 = np.min(r_f1)
                            for i in range(np.shape(r_acc)[0]):
                                logging.info(
                                    'Accuracy : %.2f Recall :%.2f F1 : %.2f N_samples : %d'
                                    % (r_acc[i], r_recall[i], r_f1[i], r_n[i]))
                            # 早停止
                            if best_accuracy >= self.params['early_stop_acc']:
                                logging.critical(
                                    'Early stop by stop_acc. Best accuracy is over {} .\
                                    Training is complete, testing the best model on x_test and y_test'.
                                    format(best_accuracy))
                                break
                        elif current_step > 300 and current_step - best_at_step >= self.params['early_stop_step']:
                            logging.critical(
                                'Early stop by stop_step. Best accuracy is over {} .\
                                Training is complete, testing the best model on x_test and y_test'.
                                format(best_accuracy))
                            break
                # Evaluate x_test and y_test
                saver.restore(sess,
                              checkpoint_prefix + '-' + str(best_at_step))
                test_batches = self.dataHelper.batch_iter(
                    list(zip(x_test, y_test)),
                    self.params['batch_size'],
                    1,
                    shuffle=False)

                total_test_correct = 0
                for test_batch in test_batches:
                    x_test_batch, y_test_batch = zip(*test_batch)
                    acc, loss, num_test_correct, predictions = dev_step(
                        x_test_batch, y_test_batch)
                    total_test_correct += int(num_test_correct)
                logging.critical('Accuracy on test set: {}'.format(
                    float(total_test_correct) / len(y_test)))
        '''保存训练的模型参数,在predict时 要load'''
        with open(trained_dir + 'words_index.json', 'w') as outfile:
            json.dump(
                self.dataHelper._vocabulary,
                outfile,
                indent=4,
                ensure_ascii=False)
        with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
            pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
        with open(trained_dir + 'labels.json', 'w') as outfile:
            json.dump(
                self.dataHelper._labels, outfile, indent=4, ensure_ascii=False)

        os.rename(path, trained_dir + 'best_model.ckpt')
        os.rename(path + '.meta', trained_dir + 'best_model.meta')
        shutil.rmtree(checkpoint_dir)
        logging.critical('{} has been removed'.format(checkpoint_dir))

        self.params['sequence_length'] = x_train.shape[1]
        with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
            json.dump(
                self.params,
                outfile,
                indent=4,
                sort_keys=True,
                ensure_ascii=False)

    def loadModel(self, trained_dir):
        '''
        加载已训练好的模型

        :param trained_dir:训练模型根目录
        :return:
        '''

        self.params = json.loads(
            open(trained_dir + 'trained_parameters.json').read())
        self.words_index = json.loads(
            open(trained_dir + 'words_index.json').read())
        self.labels = json.loads(open(trained_dir + 'labels.json').read())

        with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
            fetched_embedding = pickle.load(input_file)
        self.embedding_mat = np.array(fetched_embedding, dtype=np.float32)
        # 保存公用信息
        self.timestamp = trained_dir.split('/')[-2].split('_')[-1]
        self.trained_dir = trained_dir

        # 加载最优模型
        checkpoint_file = self.trained_dir + 'best_model.ckpt'

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                self.cnn_rnn = CNNRNN(
                    embedding_mat=self.embedding_mat,
                    non_static=bool(self.params['non_static']),
                    hidden_unit=int(self.params['hidden_unit']),
                    sequence_length=self.params['sequence_length'],
                    max_pool_size=self.params['max_pool_size'],
                    filter_sizes=map(int,
                                     self.params['filter_sizes'].split(",")),
                    num_filters=int(self.params['num_filters']),
                    num_classes=len(self.labels),
                    embedding_size=int(self.params['embedding_dim']),
                    l2_reg_lambda=float(self.params['l2_reg_lambda']))

                saver = tf.train.Saver(tf.global_variables())
                saver = tf.train.import_meta_graph(
                    "{}.meta".format(checkpoint_file[:-5]))
                saver.restore(self.sess, checkpoint_file)
                logging.critical('{} has been loaded'.format(checkpoint_file))

    def predict(self, test_x, test_y=None):
        '''
        预测

        :param test_x: 待分类文档-词集,必须
        :param test_y: 对应labels,  非必须,仅供测试用
        :return: 分类labels
        '''

        # 单文档转化为多文档形式
        # [word1,word2,...] ===> [[word1,word2,...]]
        if not (test_x and isinstance(test_x, list)):
            print('数据非法', test_x)
            return [False]
        if not isinstance(test_x[0], list):
            test_x = [test_x]

        test_x = self.dataHelper.pad_sentences(
            test_x, forced_sequence_length=self.params['sequence_length'])
        test_x = self.map_word_to_index(test_x, self.words_index)
        test_x = np.asarray(test_x)

        if test_y is not None:
            num_labels = len(self.labels)
            # 转化为one_hot向量
            one_hot = np.zeros((num_labels, num_labels), int)
            np.fill_diagonal(one_hot, 1)
            # 保存label对应one_hot向量
            label_dict = dict(zip(self.labels, one_hot))
            test_y = [label_dict[label] for label in test_y]

        # predicted_dir = '../docs/predicted_results_' + self.timestamp + '/'
        # if os.path.exists(predicted_dir):
        #     shutil.rmtree(predicted_dir)
        # os.makedirs(predicted_dir)

        batches = self.dataHelper.batch_iter(
            list(test_x), self.params['batch_size'], 1, shuffle=False)

        predictions, predict_labels = [], []
        for x_batch in batches:
            feed_dict = {
                self.cnn_rnn.input_x:
                x_batch,
                self.cnn_rnn.dropout_keep_prob:
                1.0,
                self.cnn_rnn.batch_size:
                len(x_batch),
                self.cnn_rnn.pad:
                np.zeros([len(x_batch), 1, self.params['embedding_dim'], 1]),
                self.cnn_rnn.real_len:
                self.real_len(x_batch),
            }
            predictions = self.sess.run([self.cnn_rnn.predictions], feed_dict)

            for batch_prediction in predictions[0]:
                # 索引预测值
                predictions.append(batch_prediction)
                # 真实label预测值
                predict_labels.append(self.labels[batch_prediction])

        if test_y is not None:
            test_y = np.array(np.argmax(test_y, axis=1))
            accuracy = sum(
                np.array(predictions) == test_y) / float(len(test_y))
            logging.critical('The prediction accuracy is: {}'.format(accuracy))
        # save_file = predicted_dir + 'predictions_all.txt'
        # with open(save_file,'w') as f:
        #     for label in predict_labels:
        #         f.writelines(str(label) + '\n')
        # logging.critical(
        #     'Prediction is complete, all files have been saved: {}'.format(predicted_dir))

        return predict_labels

    def map_word_to_index(self, docs_words, words_index):
        '''
        将词列表转换为index列表
        :param docs_words: 文档-词集
        :param words_index: 词-index字典
        :return:
        '''
        x_ = []
        for doc_words in docs_words:
            temp = []
            for word in doc_words:
                if word in words_index:
                    temp.append(words_index[word])
                else:
                    temp.append(0)
            x_.append(temp)
        return x_

    def real_len(self, batches):
        return [
            np.ceil((np.argmin(batch + [0]) -
                     self.params['max_pool_size'] + 1) / 2. + 1)
            for batch in batches
        ]


def pre_label(second_class):
    # 加载训练模型
    cnn_model = Text_CNN_RNN()
    cnn_model.loadModel('../docs/second_class_corpus/trained_results_{0}/'.
                        format(second_class[1]))

    # 读取数据
    corpus_raw_file = '../docs/second_class_corpus/corpus_add/%s_add.xlsx' % second_class[
        1]
    _datas, _labels = readExcelByCol(corpus_raw_file, [0, 1])
    labels = [
        _labels[i] for i in range(len(_labels))
        if _labels[i] == second_class[1]
    ]
    datas = [
        _datas[i] for i in range(len(_labels)) if _labels[i] == second_class[1]
    ]

    # 文档分词
    datas_seg = segDataset(datas)
    datas_seg = [clean_str(doc).split() for doc in datas_seg]

    # 预测类别
    pred_y = cnn_model.predict(datas_seg)
    # 保存
    import xlsxwriter
    # save
    wb = xlsxwriter.Workbook(
        "../docs/second_class_corpus/corpus_add/{0}_0620.xlsx".format(
            second_class[1]))
    ws = wb.add_worksheet()
    for row in range(len(pred_y)):
        ws.write(row, 0, datas[row])
        ws.write(row, 1, labels[row])
        ws.write(row, 2, pred_y[row])
    wb.close()


def tfidf_train(datas, labels, model_dir, tag, rate=0.95):
    tfidf = pickle.load(open("../docs/tfidf.voc", 'rb'))

    indices = np.arange(len(datas))
    np.random.shuffle(indices)
    datas = np.array(datas)[indices]
    labels = np.array(labels)[indices]
    # 文档分词
    datas_raw = segDataset(datas)
    datas = [clean_str(doc) for doc in datas_raw]
    # 切分测试集 和 训练集
    x_train, x_test, y_train, y_test = train_test_split(
        datas, labels, test_size=(1.0 - rate))
    # 转化为向量
    x_train = tfidf.transform(x_train)
    x_test = tfidf.transform(x_test)

    # 线性SVM
    svm = LinearSVC(loss='l2', penalty='l2', dual=True, tol=1e-4)
    svm.fit(x_train, y_train)
    pred = svm.predict(x_test)
    # 准确率
    score = metrics.accuracy_score(y_test, pred)
    print('accuracy: %0.3f' % score)
    # 各自准确率
    # 各自准确率
    print(metrics.classification_report(y_test, pred))
    pickle.dump(svm, open('%s/tfidf_model/SVM_%s.m' % (model_dir, tag), 'wb'))

    # SGD
    sgd = SGDClassifier(alpha=1e-4, n_iter=100, penalty='l2')
    sgd.fit(x_train, y_train)
    pickle.dump(sgd, open('%s/tfidf_model/SGD_%s.m' % (model_dir, tag), 'wb'))
    pred = sgd.predict(x_test)
    # 准确率
    score = metrics.accuracy_score(y_test, pred)
    print('accuracy: %0.3f' % score)
    # 各自准确率
    print(metrics.classification_report(y_test, pred))


if __name__ == '__main__':
    pass
