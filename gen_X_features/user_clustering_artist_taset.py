# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 25, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
from sklearn.cluster import MiniBatchKMeans, KMeans
import pandas as pd
import numpy as np
import pymysql

mysql_cn= pymysql.connect(host='localhost', port=3306,user='root', passwd='111111', db='music')

def gen_cluster(keys = None, cluster_matrix = None):
    km = MiniBatchKMeans(n_clusters=50, batch_size=1000)
#     km = KMeans(n_jobs=-1, n_clusters=50)
    print "Clustering data..."
    labels = pd.DataFrame(km.fit_predict(cluster_matrix.values))
    res = pd.concat([keys, labels], axis = 1, ignore_index=True)
    return res

def get_data():
    print "Getting data form db..."
    df = pd.read_sql('select * from user_artist_taste', mysql_cn)
    df = df.fillna(value=0)
    return df

if __name__ == "__main__":
    df = get_data()
    keys = df.pop('user_id')
    df = gen_cluster(keys, df)
    df.columns = ['user_id', 'label']
    df.to_sql('user_taste_labels', mysql_cn, flavor='mysql', if_exists='replace', 
              index = False)
    print "Wrote to db!"
    
    