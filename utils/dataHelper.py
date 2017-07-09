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
from collections import Counter, defaultdict
from multiprocessing.dummy import Pool as ThreadPool

import jieba
import jieba.posseg as pseg
import numpy as np
import word2vec
import xlrd
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfVectorizer)
from sklearn.feature_selection import SelectKBest, chi2

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
        :param test_label:测试标签集,非必须
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


class dynDict(object):
    '''
    动态字典构造词向量
    '''

    def __init__(self, X_datas, Y_labels):
        if len(X_datas) != len(Y_labels):
            print('datas is not same length as lables')
            exit()
        # 文档集
        self.X = X_datas
        # 标签集
        self.Y = Y_labels
        # 类目集
        self.cates = set(Y_labels)
        # 文本最大默认词数
        self.max_words_len = 10
        # 默认词强度
        self.theta = 1e-4
        # 语料词强度计算
        self.transform2Dict()

    def conVec(self, words):
        '''
        文本转化为词向量
        :param words_list:词列表或空格拼接的词串
        :return:词向量 长度:5*类目数
        '''
        words_vec = []
        if isinstance(words, list):
            words_list = list(set(words))
        else:
            words_list = list(set(words.split()))
        if len(words_list) > self.max_words_len:
            self.max_words_len = len(words_list)
        # 短文本最大词长
        senti_val = [self.theta] * self.max_words_len
        # 对每个类目均获取一组动态词向量
        for cate in self.cates:
            for i in range(len(words_list)):
                if words_list[i] in self.dict_senti_cate_words[cate]:
                    senti_val[i] = self.dict_senti_cate_words[cate][words_list[
                        i]]
            senti_val.sort(reverse=True)
            aver = sum(senti_val) / self.max_words_len
            top1 = senti_val[0]
            top3 = sum(senti_val[:4])
            top5 = sum(senti_val[:6])
            top7 = sum(senti_val[:8])
            # 动态词向量
            words_vec.extend([aver, top1, top3, top5, top7])

        return words_vec

    def transform2Dict(self):
        '''
        对语料进行动态词权重计算
        :return:各个类目下相对词权重
        '''
        # 各个类目下,词的分布
        dict_cate_words = dict.fromkeys(self.cates, defaultdict(int))
        # 全部语料下,词的分布
        dict_words_all = defaultdict(int)
        # 词典统计
        for i in range(len(self.Y)):
            # 文档为词列表时
            if isinstance(self.X[i], list):
                words_list = self.X[i]
            else:  # 文档为空格分割的字符串时
                words_list = self.X[i].split()
            for word in words_list:
                # 第i行语料对应的类目下该词cnt+1
                dict_cate_words[self.Y[i]][word] += 1
                # 总词典下该词cnt+1
                dict_words_all[word] += 1

        self.dict_senti_cate_words = dict.fromkeys(self.cates,
                                                   defaultdict(float))
        # 总文档数
        L = len(self.Y)
        # 各个类别分布
        cate_cnt = Counter(self.Y)
        for cate in dict_cate_words:
            # 类别cate的数目
            C = cate_cnt[cate]
            for word in dict_cate_words[cate]:
                # 全文当下词word的词频
                cnt = dict_cate_words[cate][word]
                N = dict_words_all[word]
                # 计算词权重
                weight = (cnt + 1) / (N - cnt + 1) * (L - C + 1) / (C + 1)
                # 保存
                self.dict_senti_cate_words[cate][word] = weight


def segment(content, nomial=False, only_cn=False):
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
        words = filter(is_uchar, nomial_words) if only_cn else nomial_words
        return ' '.join(nomial_words)
    else:
        words = jieba.lcut(content, HMM=True)
        words = filter(is_uchar, words) if only_cn else words
        return ' '.join(words)


def segDataset(data_set, parrel=False, nomial=False, only_cn=False):
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
            data_cut.append(segment(content, nomial, only_cn))
    end = time.time()
    print('cost time %0.2f seconds.' % (end - start))
    return data_cut


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
            ['', stop_words.add(word)][word is True]
    return stop_words


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


def clean_str(s):
    '''
    字符串清洗
    :param s:原始字符串
    :return:
    '''

    pattern = r"[0-9:()（）,!！?'`%？#.、。，+*/@“”•－：;]"
    s = re.sub(pattern, "", str(s))
    s = re.sub(r"\u2022", "", s)

    return s.strip()


def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    return True if uchar >= u'\u4e00' and uchar <= u'\u9fa5' else False


if __name__ == '__main__':
    pass
