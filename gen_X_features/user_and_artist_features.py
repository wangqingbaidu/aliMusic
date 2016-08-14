# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 30, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import datetime

featueName = 'and_artist_features'
from utils.combineTables import gen as gen_u_a_f

def gen(table_name = None):
    assert table_name    
    table_dic = {
        'user_plays_ratio_1_train':['user_id', 'artist_id', 'plays'],
        'user_plays_ratio_3_train':['user_id', 'artist_id', 'plays'],
        'user_plays_ratio_5_train':['user_id', 'artist_id', 'plays'],
        'user_plays_ratio_7_train':['user_id', 'artist_id', 'plays'],
        'user_plays_ratio_15_train':['user_id', 'artist_id', 'plays'],
        'user_plays_ratio_30_train':['user_id', 'artist_id', 'plays'],
        }
    
    return gen_u_a_f(ckeys='user_keys_train', 
                     tables = table_dic, 
                     columnCondition=['user_id', 'artist_id'], 
                     table_name=table_name)
    

def genAll():
    sql = gen('user_%s_train' %featueName)
    res = [sql, sql.replace('train', 'test'), sql.replace('train', 'submit')]
    return res
    
if __name__ == '__main__':
    for sql in genAll():
        print sql