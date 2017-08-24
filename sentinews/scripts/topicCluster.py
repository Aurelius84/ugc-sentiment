# -*- coding:utf-8 -*-
"""
Topic Cluster
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: Atom
@file: topicCluster.py
@time: 17/7/7 下午15:30
"""
import sys
from collections import defaultdict

import pandas as pd
from sklearn.cluster import AffinityPropagation

sys.path.append('/home/kevin/Documents/Project/the_machine')
from utils.dataHelper import Feature, segDataset

reload(sys)
sys.setdefaultencoding('utf8')


def xls2txt():
    df = pd.read_excel('../docs/sensitive_20.xls')
    with open('../docs/sensitive_20.txt', 'w') as f:
        for i in range(len(df['Id'])):
            cont = ['%s' % x for x in df.iloc[i, :].tolist()]
            f.write('|@|'.join(cont) + '\n')


def check():
    data = open('../docs/sensitive_20.txt', 'r').readlines()
    data = [x.split('|@|') for x in data]
    for line in data:
        try:
            assert (len(line) == len(data[0]))
        except:
            print(len(line))
            print('|@|'.join(line))
    print(len(data[0]))
    print('|@|'.join(data[0]))


if __name__ == '__main__':
    # xls2txt()
    # check()
    # exit()
    df = open('../docs/shenyang_20.txt', 'r').readlines()
    datas = [x.split('|@|') for x in df]
    desc_and_content = [line[3] + line[4] for line in datas]
    # segment
    desc_and_content_wds = segDataset(
        desc_and_content, nomial=False, only_cn=True)
    # print(desc_and_content_wds[0])
    # tfidf
    feature = Feature('../docs/stopwords.txt')
    tfidf_vec = feature.tfidf(
        desc_and_content_wds, n_features=2000, max_df=0.8, min_df=4)
    gt = feature.tfidf_vectorizer.inverse_transform
    # print(' '.join(gt(tfidf_vec[0])[0]))
    # cluster
    AP = AffinityPropagation(damping=0.9, convergence_iter=40, max_iter=500, verbose=1)
    print('start to cluster....')
    cluster_res = AP.fit(tfidf_vec)
    cluster_centers_indices = cluster_res.cluster_centers_indices_
    # cluster number
    n_clusters_ = len(cluster_centers_indices)
    # echo cluter center
    for ind in cluster_centers_indices[:10]:
        print(ind)
        print(' '.join(gt(tfidf_vec[ind])[0]))
    print(n_clusters_)
    # save by cluster
    cluter_indices = defaultdict(list)
    for line, clu in enumerate(cluster_res.labels_):
        cluter_indices[clu].append(line)
    with open('../docs/shenyang_20_cluster.txt', 'w') as f, open('../docs/shenyang_20_cluster_detail.txt', 'w') as fd:
        for key in cluter_indices:
            contents = [
                '%s|@|%s\n' % (key, ' '.join(gt(tfidf_vec[ind])[0]))
                for ind in cluter_indices[key]
            ]
            f.writelines(contents)

            detail = ['%s|@|%s' % (key, df[ind]) for ind in cluter_indices[key]]
            fd.writelines(detail)
