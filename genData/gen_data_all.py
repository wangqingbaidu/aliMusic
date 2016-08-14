# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 21, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
from genData.X_train import gen as gen_X_train
from genData.X_test import gen as gen_X_test
from genData.X_submit import gen as gen_X_submit
from genData.y_train import gen as gen_y_train
from genData.y_test import gen as gen_y_test

def genAll(keys_type = None, ignore_date = True):
    return genAll_X() + genAll_y(keys_type, ignore_date)

def genAll_X():
    return [gen_X_train('user_keys_train', ['user_id'],'user_X_train'),
            gen_X_test('user_keys_test', ['user_id'],'user_X_test'),
            gen_X_submit('user_keys_submit', ['user_id'],'user_X_submit'),]
    
def genAll_y(keys_type = None, ignore_date = True):
    assert keys_type
    conditions = None
    ds = []
    if not ignore_date:
        ds = ['ds']
    if keys_type == 'u_a_d':
        conditions = ['user_id', 'artist_id'] + ds
    elif keys_type == 'u_s_d':
        conditions = ['user_id', 'song_id'] + ds
    return [gen_y_train('user_keys_train', conditions,'user_y_train'),
            gen_y_test('user_keys_test', conditions,'user_y_test')]

if __name__ == '__main__':
    for s in genAll_X():
        print s