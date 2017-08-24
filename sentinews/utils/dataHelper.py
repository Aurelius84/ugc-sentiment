# -*- coding:utf-8 -*-
"""
数据预处理模块
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: Atom
@file: data-help.py
@time: 17/7/7 下午15:21
"""
import logging
import re
import time
from math import ceil
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool

import jieba
import jieba.posseg as pseg
import numpy as np
import word2vec
import xlrd
from sklearn.feature_extraction.text import (
    CountVectorizer, HashingVectorizer, TfidfVectorizer)
from sklearn.feature_selection import SelectKBest, chi2
import os
import itertools
from os.path import join, exists, split
logging.getLogger().setLevel(logging.INFO)


class Feature(object):
    """
    特征抽取抽象类
    支持一下词表示:
        1.TF词频 表示词向量
        2.TF-IDF 表示词向量
        3.CHI 表示词权重向量
        4.Hash map
        5.Word2Vec 累加表示词向量
    """

    def __init__(self, stop_words_file):
        '''
        初始化
        :param stop_words_file: 停用词路径
        :return:
        '''
        # 停用词表
        self.stop_words = self.stopwords(stop_words_file)

    def tf(self, X_data, test_data=None, n_features=30000):
        '''
        tf词频
        :param X_data:训练文档集,必须
        :param test_data:测试文本或文本集,非必须
        :param n_features:特征数 top_n原则
        :return:特征向量
        '''
        # 局部init
        try:
            if self.tf_vectorizer:
                pass
        except:
            self.tf_vectorizer = CountVectorizer(
                max_features=n_features,
                stop_words=self.stop_words,
                decode_error='strict')
            x_data_vec = self.tf_vectorizer.fit_transform(X_data)
            print("vocabulary length is %s:" %
                  len(self.tf_vectorizer.vocabulary_))

        # 只对测试文本处理时
        if test_data is not None:  # 支持处理单条或多条测试文本
            return self.tf_vectorizer.transform(test_data)

        # 无测试文本时,返回训练集向量
        return x_data_vec

    def tfidf(self,
              X_data,
              test_data=None,
              n_features=30000,
              max_df=0.95,
              min_df=2):
        '''
        tfidf 词向量统计
        :param X_data:训练文本集,必须
        :param test_data:测试文本或文本集,非必须
        :param n_features:最大特征数
        :param max_df:
        :param min_df:
        :return:词向量
        '''
        # 局部init
        try:
            if self.tfidf_vectorizer:
                pass
        except:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=n_features,
                stop_words=self.stop_words,
                max_df=max_df,
                min_df=min_df)
            x_data_vec = self.tfidf_vectorizer.fit_transform(X_data)
            print("vocabulary length is %s:" %
                  len(self.tfidf_vectorizer.vocabulary_))

        # 只对测试文本处理时
        if test_data is not None:  # 支持处理单条或多条测试文本
            return self.tfidf_vectorizer.transform(test_data)

        # 无测试文本时,返回训练集向量
        return x_data_vec

    def chi(self, X_data, Y_data, test_data=None, n_features=2000):
        '''
        CHI词向量表示
        :param X_data:训练文本集,必须
        :param Y_data:训练标签集,必须
        :param test_data:测试文本集,非必须
        :param test_label:测试标签集,非必须load_data
        :param n_features:特征数
        :return:特征向量
        '''
        try:
            if self.chi_2:
                pass
        except:
            # 先转化为tfidf向量矩阵
            self.chi_tfidf_vectorizer = TfidfVectorizer(
                stop_words=self.stop_words)
            x_data_tfidf_vec = self.chi_tfidf_vectorizer.fit_transform(X_data)
            print("vocabulary length is %s:" %
                  len(self.chi_tfidf_vectorizer.vocabulary_))

            # 卡方
            self.chi_2 = SelectKBest(chi2, k=n_features)
            x_data_vec = self.chi_2.fit_transform(x_data_tfidf_vec, Y_data)

        # 只对测试文本处理时
        if test_data is not None:  # 支持处理单条或多条测试文本
            test_data_tfidf_vec = self.chi_tfidf_vectorizer.transform(
                test_data)
            return self.chi_2.transform(test_data_tfidf_vec)

        # 无测试文本时,返回训练集向量
        return x_data_vec

    def hash(self, X_data, n_features=2**18):
        '''
        hash 特征向量
        :param X_data: u'content','file','list','filename'
        :param n_features: 特征数目 默认2 ** 18
        :return: 特征向量
        '''
        # hash特征
        vectorizer = HashingVectorizer(
            stop_words=self.stop_words,
            non_negative=True,
            n_features=n_features)
        X_data_vec = vectorizer.transform(X_data)

        return X_data_vec

    def stopwords(self, file):
        '''
        加载停用词表
        :param file:停用词表路径
        :return:停用词列表
        '''
        with open(file, 'r') as f:
            stop_words = list(f.read().splitlines())

        return stop_words

    def wordToVec(self, X_data, model_file='../docs/txtAll.bin'):
        '''
        word2vec 词向量表示
        :param X_data:文档集
        :param model_file:bin文件目录
        :return:词向量
        '''
        # 文档词向量
        x_data_vec = []
        # 局部初始化
        try:
            if self.model:
                pass
        except:
            # Word2Vec向量训练集
            self.model = word2vec.load(model_file)
        for doc in X_data:
            words = segmentByStopwords(doc, self.stop_words)
            # 句子向量
            sent_vec = np.zeros([1, 100])
            for word in words:
                sent_vec += wordToVec(word, self.model)
            x_data_vec.append(sent_vec)

        return x_data_vec


def train_word2vec(sentence_matrix,
                   vocabulary_inv,
                   num_features=100,
                   min_word_count=1,
                   context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    model_dir = os.path.abspath('../docs') + '/model/w2v_matrix'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(
        num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(
            sentences,
            workers=num_workers,
            size=num_features,
            min_count=min_word_count,
            window=context,
            sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using
        # Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = [
        np.array([
            embedding_model[w] if w in embedding_model else np.random.uniform(
                -0.25, 0.25, embedding_model.vector_size)
            for w in vocabulary_inv
        ])
    ]
    return embedding_weights


def load_data_and_labels(file_path,
                         split_tag='\t',
                         lbl_text_index=[0, 1],
                         use_jieba_segment=False,
                         is_shuffle=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # Load data from files
    raw_data = list(open(file_path, 'r').readlines())
    # parse label
    labels = [
        data.strip('\n').split(split_tag)[lbl_text_index[0]]
        for data in raw_data
    ]
    # parse text
    texts = [
        data.strip('\n').split(split_tag)[lbl_text_index[1]]
        for data in raw_data
    ]
    if use_jieba_segment:
        texts = segDataset(texts)

    # Split by words
    # texts = [clean_str(sent) for sent in texts]
    texts = [filter(lambda a: a != '', s.split(" ")) for s in texts]
    # support multi-label
    labels = [filter(lambda a: a != '', s.split(" ")) for s in labels]
    if is_shuffle:
        ind = np.arange(len(texts))
        np.random.shuffle(ind)
        texts = list(np.array(texts)[ind])
        labels = list(np.array(labels)[ind])

    return texts, labels


def load_trn_tst_data_labels(trn_file,
                             tst_file=None,
                             ratio=0.2,
                             split_tag='\t',
                             lbl_text_index=[0, 1],
                             is_shuffle=False):
    """
    Loads train data and test data,return segment words and labels
    if tst_file is None , split train data by ratio
    """
    # train data
    trn_data, trn_labels = load_data_and_labels(
        trn_file, split_tag, lbl_text_index, is_shuffle=is_shuffle)

    # test data
    if tst_file:
        tst_data, tst_labels = load_data_and_labels(
            tst_file, split_tag, lbl_text_index, is_shuffle=is_shuffle)
    else:
        index = np.arange(len(trn_labels))
        np.random.shuffle(index)
        split_n = int(ratio * len(trn_labels))

        trn_data = np.array(trn_data)
        trn_labels = np.array(trn_labels)

        tst_data, tst_labels = trn_data[index[:split_n]], trn_labels[
            index[:split_n]]
        trn_data, trn_labels = trn_data[index[split_n:]], trn_labels[index[
            split_n:]]

    return list(trn_data), list(trn_labels), list(tst_data), list(tst_labels)


def pad_sentences(sentences, padding_word="<PAD/>", mode='max'):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if isinstance(mode, int):
        sequence_length = mode
    elif mode == 'max':
        sequence_length = max(len(x) for x in sentences)
    else:
        sequence_length = sum(len(x) for x in sentences) / len(sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences, padding_word="<PAD/>"):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # set padding_word id == 0
    vocabulary_inv.remove(padding_word)
    vocabulary_inv.insert(0, padding_word)
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary, padding_word="<PAD/>"):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[
        vocabulary[word] if word in vocabulary else vocabulary[padding_word]
        for word in sentence
    ] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(trn_file,
              tst_file=None,
              ratio=0.2,
              split_tag='\t',
              lbl_text_index=[0, 1],
              vocabulary=None,
              vocabulary_inv=None,
              padding_mod='max',
              is_shuffle=True,
              use_jieba_segment=False,
              use_tst=False):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    print("%s  loading train data and label....." %
          time.asctime(time.localtime(time.time())))
    trn_text, trn_labels = load_data_and_labels(
        trn_file,
        split_tag,
        lbl_text_index,
        is_shuffle=is_shuffle,
        use_jieba_segment=use_jieba_segment)
    if tst_file:
        print("%s  loading train data and label....." %
              time.asctime(time.localtime(time.time())))
        tst_text, tst_labels = load_data_and_labels(
            tst_file, split_tag, lbl_text_index, is_shuffle=is_shuffle)
        sentences, labels = trn_text + tst_text, trn_labels + tst_labels
    else:
        sentences, labels = trn_text, trn_labels
    print("%s  padding sentences....." %
          time.asctime(time.localtime(time.time())))
    sentences_padded = pad_sentences(sentences, mode=padding_mod)

    if vocabulary is None or vocabulary_inv is None:
        print("%s  building vocab....." %
              time.asctime(time.localtime(time.time())))
        vocabulary, vocabulary_inv = build_vocab(sentences_padded)

    x, y = build_input_data(sentences_padded, labels, vocabulary)

    if tst_file is None and not use_tst:
        return [x, y, vocabulary, vocabulary_inv]
    elif tst_file:
        split_n = len(trn_text)
    elif use_tst:
        split_n = int(ratio * len(trn_text))

    return x[split_n:], y[
        split_n:], x[:split_n], y[:split_n], vocabulary, vocabulary_inv


def readExcelByCol(file_name, col):
    '''
    按列读取excel数据
    :param file_name: excel路径
    :param col: 列数,可以多列,或者单列
    :return: 迭代器 可调用next(gener)来返回数据
    '''
    wb = xlrd.open_workbook(file_name)
    sheet = wb.sheet_by_index(0)

    if isinstance(col, list):
        for i in col:
            yield sheet.col_values(i)
    elif isinstance(col, int):
        data = sheet.col_values(col)
        yield data


def corpus_balance(datas, labels, mod='max'):
    """对语料进行重采样 或 欠采样."""
    shuffle_indices = np.random.permutation(np.arange(len(datas)))
    shuffle_datas = np.array(datas)[shuffle_indices]
    shuffle_labels = np.array(labels)[shuffle_indices]
    keys = set(labels)
    label_count = Counter(labels)
    # 以语料 最多 为标准，对其他类别进行 过采样
    if mod == 'max':
        max_label = label_count.most_common(1)[0][0]
        keys.remove(max_label)
        for key in keys:
            data = shuffle_datas[shuffle_labels == key].tolist()
            label = shuffle_labels[shuffle_labels == key].tolist()
            ratio = int((label_count[max_label]) * 1. / label_count[key] - 1.5)
            if ratio < 0:
                continue
            for _ in range(ratio):
                datas.extend(data)
                labels.extend(label)
    # 以语料 最少 为标准，对其他类别进行 欠采样
    elif mod == 'min':
        min_label = label_count.most_common()[-1][0]
        datas = shuffle_datas[shuffle_labels == min_label].tolist()
        labels = shuffle_labels[shuffle_labels == min_label].tolist()
        keys.remove(min_label)
        for key in keys:
            data = shuffle_datas[shuffle_labels == key][:label_count[
                min_label]].tolist()
            label = shuffle_labels[shuffle_labels == key][:label_count[
                min_label]].tolist()
            datas.extend(data)
            labels.extend(label)
    # 均衡采样， 对最多的进行欠抽样，对最少的进行重采样
    elif mod == 'average':
        ind = int((len(keys) - 1) / 2.)
        average_label = label_count.most_common()[ind][0]
        datas = shuffle_datas[shuffle_labels == average_label].tolist()
        labels = shuffle_labels[shuffle_labels == average_label].tolist()
        keys.remove(average_label)
        for key in keys:
            ratio = label_count[key] * 1. / label_count[average_label]
            data = shuffle_datas[shuffle_labels == key]
            label = shuffle_labels[shuffle_labels == key]
            if ratio < 0.8:
                data_shape = (
                    int(1. / ratio + 0.2),
                    1) if len(data.shape) == 2 else int(1. / ratio + 0.2)
                label_shape = (
                    int(1. / ratio + 0.2),
                    1) if len(label.shape) == 2 else int(1. / ratio + 0.2)
                data = np.tile(data, data_shape)
                label = np.tile(label, label_shape)
            elif ratio > 1.2:
                ind = int((1. / ratio + 0.2) * len(data))
                data = data[:ind]
                label = label[:ind]
            datas += data.tolist()
            labels += label.tolist()

    return datas, labels


def segment(content, nomial=False):
    '''
    分词
    :param content: 待分词文本
    :param nomial:是否仅仅保留动名词
    :return:'a b c'
    '''
    content = str(content)
    if nomial:
        nomial_words = []
        words = pseg.cut(content)
        for word in words:
            # print(word.word,word.flag)
            if contain(['n', 'v', 'd', 'a'], word.flag):
                nomial_words.append(word.word)
        return ' '.join(nomial_words)
    else:
        words = jieba.lcut(content.strip('\n'), HMM=True)
        return ' '.join(words)


def segDataset(data_set, parrel=False, nomial=False):
    '''
    文档集分词
    :param data_set:文档集
    :param parrel:是否并行分词,windows不支持
    :param nomial:是否仅仅保留动名词
    :return:['a b c','e d g',....]
    '''
    data_cut = []
    start = time.time()
    print('start cut dataset....')
    if parrel:
        p = ThreadPool(4)
        data_cut = p.map(segment, data_set)
        p.close()
        p.join()
    else:
        for content in data_set:
            data_cut.append(segment(content, nomial))
    end = time.time()
    print('cost time %0.2f seconds.' % (end - start))
    return data_cut


def segmentByStopwords(sentence, stop_words):
    '''
    :param sentence:
    :return:
    '''
    words = jieba.lcut(sentence, cut_all=False)
    for _ in range(len(words)):
        word = words.pop(0)
        if word not in stop_words:
            words.append(word)
    return words


def loadStopWords(file_name):
    '''
    加载停用词
    :param file_name: 停用词表路径
    :return: set()
    '''
    stop_words = set()
    with open(file_name, 'rb') as f:
        for line in f:
            word = line.decode('utf-8').strip()
            ['', stop_words.add(word)][word != '']
    return stop_words


def clean_str(s):
    '''
    字符串清洗
    :param s:原始字符串
    :return:
    '''

    pattern = r"[~（）&:,!！﹗?'`%？#.、。，+*/“”•－：;；·—．-]"
    s = re.sub(r"#.*#|＃.*＃", "", str(s))
    s = re.sub(r"【.*?】", "", s)
    s = re.sub(r"@.*?@", "", s)
    s = re.sub(r"[0-9]+", "Number", s)
    s = re.sub(r"\(\)|（）|__*", "Blank", s)
    s = re.sub(pattern, "", s)
    s = re.sub("nbsp;|amp;|quot;|qpos;", "", s)
    s = re.sub(r"\u2022", "", s)
    s = re.sub(r"大同区同福社区|青年之声", "", s)

    return s.strip()


def save_var(file_path, var):
    with open(file_path, 'w') as f:
        var = ['%s\n' % x for x in var]
        f.writelines(var)


def load_var(file_path):
    with open(file_path, 'r') as f:
        var = [x.decode('utf8').strip('\n') for x in f.readlines()]
        return var


def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    return True if uchar >= u'\u4e00' and uchar <= u'\u9fa5' else False


def contain(_list, _string):
    '''
    检测字符串是否包含某种字符
    :param _list: 字符模式
    :param _string: 匹配字符串
    :return: T or F
    '''
    for _str in _list:
        if _string.find(_str) != -1:
            return True
    return False


def wordToVec(word, model):
    '''
    获取词向量
    :param word: utf-8格式 词
    :return: 词向量一维矩阵
            [0.1,0.3,....]
    '''
    try:
        return model[word]
    except KeyError:
        print('no key of {0}'.format(word))
        return np.zeros([1, 100])


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


class cnn_data_helper(object):
    """
    CNN_LSTM 专属的数据清洗流程
    流程包括如下:
            1.labels对角向量化
            2.构造labels映射字典
            3.原始文本padding长度fixing处理
            4.文本向量化
    """

    def __init__(self, word2vec_model_dir='../docs/sentiment/extend_dict.txt'):
        '''
        初始化

        :param word2vec_model_dir: word2vec训练的bin文件路径
        :return:
        '''

        self.word2vec_model = loadSentimentVector(word2vec_model_dir)
        self.vocabulary_inv = self.word2vec_model.keys()
        self.vocabulary_inv = ["<PAD/>"] + self.vocabulary_inv
        self.vocabulary = {
            word: ind
            for ind, word in enumerate(self.vocabulary_inv)
        }

    def fit(self, train_x_raw, train_y_raw, padding_mod='max'):
        '''
        训练集空间向量化

        :param train_x_raw:训练文档-词集
        :param train_y_raw:训练文档标签集
        :return:
        '''

        # labels集合
        self.labels = list(set(train_y_raw))
        num_labels = len(self.labels)
        # 转化为one_hot向量
        one_hot = np.zeros((num_labels, num_labels), int)
        np.fill_diagonal(one_hot, 1)
        # 保存label对应one_hot向量
        self.label_dict = dict(zip(self.labels, one_hot))

        # self.build_vocab(train_x_raw)
        # x 是文档-词向量矩阵 固定维度
        train_x_raw = [[
            self.vocabulary[word] for word in doc_words
            if word in self.vocabulary
        ] for doc_words in train_x_raw]
        train_x = self.pad_sentences(
            train_x_raw, padding_word=0, forced_sequence_length=padding_mod)
        y_raw = [self.label_dict[label] for label in train_y_raw]

        train_x = np.array(train_x)
        train_y = np.array(y_raw)

        return train_x, train_y

    def load_embeddings(self, dim=100):
        '''
        加载词典embedding向量

        :return:
        '''
        word_embeddings = {}
        for word in self.vocabulary:
            try:
                word_embeddings[word] = self.word2vec_model[word]
            except:
                word_embeddings[word] = np.random.uniform(-0.1, 0.1, dim)
        return word_embeddings

    def pad_sentences(self,
                      docs_words,
                      padding_word="<PAD/>",
                      forced_sequence_length='average'):
        '''
        归整文档词序列长度

        :param docs_words:文档-词集
        :param padding_word: 默认添加的词
        :param forced_sequence_length: 训练时制定的词序列长度, int or str 'max' and 'average'
        :return: pad 后的文档-词集
        '''
        """Pad setences during training or prediction"""
        if forced_sequence_length == 'average':
            sequence_length = [len(doc_word) for doc_word in docs_words]
            self.sequence_length = int(
                1.3 * sum(sequence_length) / len(sequence_length))

        elif forced_sequence_length == 'max':
            self.sequence_length = max(
                len(doc_word) for doc_word in docs_words)
        else:
            self.sequence_length = int(forced_sequence_length)

        # 存放 pad后的docs
        padded_sentences = []
        for doc_index in range(len(docs_words)):
            doc_words = docs_words[doc_index]
            num_padding = self.sequence_length - len(doc_words)

            if num_padding < 0:  # Prediction: cut off the sentence if it is longer than the sequence length
                # logging.info(
                #     'This sentence has to be cut off because it is longer than trained sequence length')
                padded_sentence = doc_words[0:self.sequence_length]
            else:
                # 训练语句长度不够的话,补全
                padded_sentence = doc_words + [padding_word] * num_padding
            padded_sentences.append(padded_sentence)
        return padded_sentences

    def build_vocab(self, train_x_raw_padded):
        '''
        构建总词典集词典词序映射表

        :param train_x_raw_padded: 文档-fixed词集
        :return:
        '''

        word_counts = Counter(itertools.chain(*train_x_raw_padded))
        self.vocabulary_inv = [word[0] for word in word_counts.most_common()]
        self.vocabulary = {
            word: index
            for index, word in enumerate(self.vocabulary_inv)
        }

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        '''
        数据batch随机化

        :param data: list(zip(x_train, y_train))
        :param batch_size:
        :param num_epochs:
        :param shuffle:
        :return:
        '''
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(ceil(float(data_size) / batch_size))

        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def transforms(self,
                   test_x_raw,
                   padding_word="<PAD/>",
                   forced_sequence_length=None):
        '''
        测试文本向量化
        :param test_x_raw: 测试文本 str or list
        :return:
        '''
        if not isinstance(test_x_raw[0], list):
            test_x_raw = [test_x_raw]

        test_x_raw = self.pad_sentences(
            test_x_raw,
            padding_word=padding_word,
            forced_sequence_length=forced_sequence_length)
        # x 是文档-词向量矩阵 固定维度
        train_x = np.array([[self.vocabulary[word] for word in doc_words]
                            for doc_words in test_x_raw])

        return train_x

    @property
    def _vocabulary(self):
        return self.vocabulary

    @property
    def _vocabulary_inv(self):
        return self.vocabulary_inv

    @property
    def _labels(self):
        return self.labels

    @property
    def _sequence_length(self):
        return self.sequence_length


if __name__ == '__main__':
    pass
