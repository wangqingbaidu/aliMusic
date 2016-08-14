# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 23, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import pymysql

def calc_user_runoff():
    mysql_cn= pymysql.connect(host='10.41.0.63', port=3306,user='root', passwd='111111', db='music')
    before_df = pd.read_sql('''
    SELECT user_id from user_actions
    WHERE ds >='20150301' and ds <= '20150430'
    GROUP BY user_id''', mysql_cn).values
    before_df = set(before_df.reshape(before_df.shape[0]).tolist())
    
    
    after_df = pd.read_sql('''
    SELECT user_id from user_actions
    WHERE ds >='20150401' and ds <= '20150530'
    GROUP BY user_id''', mysql_cn).values
    after_df = set(after_df.reshape(after_df.shape[0]).tolist())
    
    
    future_df = pd.read_sql('''
    SELECT user_id from user_actions
    WHERE ds >='20150601' and ds <= '20150830'
    GROUP BY user_id''', mysql_cn).values
    
    future_df = set(future_df.reshape(future_df.shape[0]).tolist())
    
    print 'first two month b:%d a:%d diff:%d' %(len(before_df), len(after_df), len(before_df & after_df))
    print 'later two month a:%d f:%d diff:%d' %(len(after_df), len(future_df), len(after_df & future_df))

def calc_user_taste_change():
    mysql_cn= pymysql.connect(host='10.41.0.63', port=3306,user='root', passwd='111111', db='music')
    before_df = pd.read_sql('''
    SELECT user_id, artist_id from user_keys_train
    GROUP BY user_id, artist_id''', mysql_cn).values
    before_df = set([i[0] +i[1] for i in before_df.tolist()])

    after_df = pd.read_sql('''
    SELECT user_id, artist_id from user_keys_test
    GROUP BY user_id, artist_id''', mysql_cn).values
    after_df = set([i[0] +i[1] for i in after_df.tolist()])
    
    future_df = pd.read_sql('''
    SELECT user_id, artist_id from user_keys_submit
    GROUP BY user_id, artist_id''', mysql_cn).values
    future_df = set([i[0] +i[1] for i in future_df.tolist()])
    
    print 'first two month b:%d a:%d diff:%d' %(len(before_df), len(after_df), len(before_df & after_df))
    print 'later two month a:%d f:%d diff:%d' %(len(after_df), len(future_df), len(after_df & future_df))
    
if __name__ == '__main__':
    calc_user_taste_change()
