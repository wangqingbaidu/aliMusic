# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 28, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import pymysql
import matplotlib                                                                        
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import datetime

def xx(x):
    return 1.0/(1+np.log2(x+0.12)/np.log2(100))
    
def new_songs_plays():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='music')
    artist_list = pd.read_sql('select album, plays from album_list order by album desc', 
                              mysql_cn).values.tolist()
    album_list = [(x[0], x[1]) for x in artist_list]
    count = 0
    X_train = None
    y_train  = None
    first = False 
    X_test = []
    y_test = []
    inverse = []
    album_set = []
    yt_index = []
    for album, plays in album_list:
        df = pd.read_sql('''
        SELECT plays from new_songs_plays
        WHERE album = '{album}' and ds >= '{start}'
        order by ds;
        '''.format(album=album, start = album[-8:]),mysql_cn)
        if np.max(df.values) < 1000:
            continue
        
        y_index = df.values.argmax()
        y = df.astype(float).values[y_index:]
        ss=MinMaxScaler(feature_range=(0.1, 0.9))
        ss=ss.fit(y)
        y=ss.transform(y)
        y = y.reshape((y.shape[0]))
        X = np.arange(1, y.shape[0] + 1)
        X = X.reshape((-1, 1))
#         if random.randint(0, 10) <= 1 and X.shape[0] <= 30:
        if album[-8:] >= '20150701':
            y_test.append(y)
            X_test.append(X)
            inverse.append(ss)
            album_set.append(album)
            yt_index.append(y_index)
        else:
            if not first:
                y_train = y
                X_train = X
                first = True
            else:
                y_train = np.hstack((y_train, y))
                X_train = np.vstack((X_train, X))
                
    lr = LinearRegression()
    X_train=xx(X_train)

    lr.fit(X_train, y_train)
    for xt, yt, ss in zip(X_test, y_test, inverse):
        df = pd.DataFrame(yt)
        res= lr.predict(xx(xt))
        #res = 1 / (np.exp(lr.predict(xt)) + 1)

        df['predict'] = pd.DataFrame(res)
        df.columns = ['origin', 'predict']
        df.plot()
        plt.title('%s' %album)
        fig = plt.gcf()
        fig.savefig('./img/nlog_No{No:0>3}.png'.format(No = count))
        plt.close()
        count += 1
        
    X_all = np.arange(1, 365)
    X_all = X_all.reshape((-1, 1))   
    result = []
    for album, ss, index in zip(album_set,inverse, yt_index):
        gapday = datetime.timedelta(days=index) 
        dateFrom = datetime.datetime.strptime(album[-8:], '%Y%m%d')
        if dateFrom.strftime('%Y%m%d') >= '20150701':
            continue
        print album, index
        pred = ss.inverse_transform(lr.predict(xx(X_all)).reshape((-1,1))).reshape((-1))
#         pred = pred - np.max(pred)
        for plays in pred.tolist():
            dateNow = (dateFrom + gapday).strftime('%Y%m%d')
            gapday += datetime.timedelta(days=1) 
            if dateNow > '20151030':
                break
            result.append([album[:-8], dateNow, float(plays)])
    result = pd.DataFrame(result)
#     print result
    result.columns = ['artist_id', 'ds', 'plays']
#     result.to_sql('new_songs_decay', mysql_cn, flavor='mysql', if_exists='replace', index = False)
    result.to_csv('./new_songs_incr.csv', index=False)
    mysql_cn.close()
    
if __name__ == '__main__':
    new_songs_plays()

'''
47c05597b30c1fc870d2dba43e318fdb        
        
        
'''