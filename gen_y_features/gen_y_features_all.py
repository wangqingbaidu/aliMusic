# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 13, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
from gen_y_features import user_artist_y_features_test, user_artist_y_features_train,\
    user_song_y_features_test, user_song_y_features_train
def gen_y_features_all(keys_type = None, ignore_date = True):
    assert keys_type
    if keys_type == 'u_a_d':
        return [user_artist_y_features_train.gen(ignore_date=ignore_date), 
                user_artist_y_features_test.gen(ignore_date=ignore_date)]
    elif keys_type == 'u_s_d':
        return [user_song_y_features_train.gen(ignore_date=ignore_date), 
                user_song_y_features_test.gen(ignore_date=ignore_date)]
        


if __name__ == '__main__':
    sql = ''
    for s in gen_y_features_all('u_a_d', True):
        sql += s
    print sql