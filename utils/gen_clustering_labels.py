# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 25, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np

def gen_cluster(keys = None, cluster_matrix = None):
    assert cluster_matrix and keys
    km = MiniBatchKMeans(n_clusters=50, batch_size=1000)
    labels = pd.DataFrame(km.fit_predict(cluster_matrix.values))
    
    res = pd.concat([keys, labels], axis = 1, ignore_index=True)
    return res