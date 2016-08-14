# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 15, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''

from utils.combineTables import gen as gen_X_submit
from genData import totalArtists, language_set

def gen(ckeys = None, conditions = None, table_name = None):
    assert ckeys and conditions, table_name
    fav_hour_columns = ["fav_hour{index}".format(index = '{:0>2}'.format(index))
        for index in range(24)]
    fav_artist_columns = ['fav_artist{index}'.format(index = index + 1) 
        for index in range(totalArtists)]
    fav_gender_columns = ['gender1','gender2','gender3']
    
    fav_language_columns = ["fav_lan{l}".format(l = l) 
                                 for l in language_set]
    table_dic = {
        'user_taste_labels':['user_id', 'label'],
        'user_day_avg_play_submit':['user_id', 'day_avg'], 
        'user_day_avg_download_submit':['user_id', 'day_avg'], 
        'user_day_avg_collect_submit':['user_id', 'day_avg'],
        'user_favorite_hour_submit':['user_id'] + fav_hour_columns, 
        'user_favorite_artist_submit':['user_id'] + fav_artist_columns, 
        'user_favorite_gender_submit':['user_id'] + fav_gender_columns,
        'user_favorite_language_submit':['user_id'] + fav_language_columns,
        }
    
    return gen_X_submit(ckeys=ckeys, tables = table_dic, columnCondition=conditions,table_name=table_name)
    
if __name__ == '__main__':
    print gen('user_keys_submit', ['user_id'],'user_X_submit')
