# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 16, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import pymysql
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time, os

#if __name__ == '__main__':
def get_train_predict(model = None):
    mysql_cn= pymysql.connect(host='localhost', port=3306,user='root', passwd='111111', db='music')
    
    print "Model %s Got!" %type(model)
     
    print "Getting X train data"
    X_train = pd.read_sql('select * from user_X_train;', con=mysql_cn)
    X_train = X_train.fillna(value=0).values
    
    print "Getting y train data"
    y_train = pd.read_sql('select * from user_y_train;', con=mysql_cn) 
    y_train = y_train.fillna(value=0).values
    y_train = y_train.reshape((y_train.shape[0]))
    
    print "Getting X test data"
    X_test= pd.read_sql('select * from user_X_test;', con=mysql_cn)     
    X_test= X_test.fillna(value=0).values
    
    print 'Fitting data...'
    model.fit(X_train, y_train)
    
    print 'Predicting data...'
    y_test = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    
    print "Getting test keys"
    keys_test = pd.read_sql('select * from user_keys_test;', con=mysql_cn)
    
    res = pd.concat([keys_test, y_test], axis = 1, ignore_index=True)
    
    mysql_cn.close()
    return res

def get_test_predict(model = None):
    mysql_cn= pymysql.connect(host='localhost', port=3306,user='root', passwd='111111', db='music')
    
    print "Model %s Got!" %type(model)
    
    print "Getting X test data"
    X_train = pd.read_sql('select * from user_X_test;', con=mysql_cn)
    X_train = X_train.fillna(value=0).values
    
    print "Getting y test data"
    y_train = pd.read_sql('select * from user_y_test;', con=mysql_cn) 
    y_train = y_train.fillna(value=0).values
    y_train = y_train.reshape((y_train.shape[0]))
    
    print "Getting X submit data"
    X_test= pd.read_sql('select * from user_X_submit;', con=mysql_cn)     
    X_test= X_test.fillna(value=0).values
    
    print 'Fitting data...'
    model.fit(X_train, y_train)
    
    print 'Predicting data...'
    y_test = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    
    print "Getting submit keys"
    keys_test = pd.read_sql('select * from user_keys_submit;', con=mysql_cn)
    
    res = pd.concat([keys_test, y_test], axis = 1, ignore_index=True)
    
    mysql_cn.close()
    return res
 
def gen_predic_csv(dateNow = time.strftime('%Y%m%d'), timeNow = time.strftime('%H%M%S')):
    model = RandomForestRegressor(n_jobs=-1)
    get_train_predict(model).to_csv('./%s/%s_train.csv' %(dateNow, timeNow), 
                                    header=False, index = False)
    
    get_test_predict(model).to_csv('./%s/%s_test.csv' %(dateNow, timeNow), 
                                    header=False, index = False) 
    
def get_songDic(ifile = None):
    assert ifile
    songDic = {}
    f = open(ifile).readlines()
    for item in f:
        items = item.split(',')
        if not songDic.has_key(items[0]):
            songDic[items[0]] = items[1]
    return songDic
    
def get_artist(ifile = None, songdic= None):
    assert ifile and songdic
    f = open(ifile).readlines()
    res = []
    for item in f:
        items = item.split(',')
        if songdic.has_key(items[1]):
            if items[2] == '20151031' or items[2] == '20150831':
                continue
            res.append([songdic[items[1]], float(items[3].split('\n')[0]), items[2]])
    return res

def gen_result_csv():
    dateNow = time.strftime('%Y%m%d')
    timeNow = time.strftime('%H%M%S')
    if not os.path.exists(dateNow):
        os.system('mkdir %s' %dateNow)
    print '-------------------------Getting model...---------------------------------'
    gen_predic_csv(dateNow, timeNow)
    print '---------------------------Model got!-------------------------------------'
    
    print 'Getting artist_song dic...'
    songDic = get_songDic('songs_artist.csv')
    
    print '-----------------------Getting train results------------------------------'
    print 'Getting dataframe of train data...'
    df = pd.DataFrame(np.asarray(get_artist('./%s/%s_train.csv' %(dateNow, timeNow), songDic)), 
                      columns=['artist_id','plays', 'ds'])
    df['plays'] = df['plays'].astype(float)
    df = df.groupby(['artist_id', 'ds']).sum()    
    df['plays'] = df['plays'].astype(int)
    df = df.reset_index()
    ds = df.pop('ds')
    df.insert(2, 'ds', ds)
    print 'Saving train results...'
    df.to_csv('./%s/%s_train_results.csv' %(dateNow, timeNow), header =False, index=False)
    
    
    print '-----------------------Getting test results--------------------------------'
    print 'Getting dataframe of test data...'
    df = pd.DataFrame(np.asarray(get_artist('./%s/%s_test.csv' %(dateNow, timeNow), songDic)), 
                      columns=['artist_id','plays', 'ds'])
    df['plays'] = df['plays'].astype(float)
    df = df.groupby(['artist_id', 'ds']).sum()  
    df['plays'] = df['plays'].astype(int)  
    df = df.reset_index()
    ds = df.pop('ds')
    df.insert(2, 'ds', ds)
    print 'Saving test results...'
    df.to_csv('./%s/%s_test_results.csv' %(dateNow, timeNow), header =False, index=False)
    
if __name__ == '__main__':
    gen_result_csv()
