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
import matplotlib.pyplot as plt

def calc_artist_miss():
    mysql_cn= pymysql.connect(host='10.41.0.63', port=3306,user='root', passwd='111111', db='music')
    res_list = [x[0] 
                for x in pd.read_sql('select artist_id from songs group by artist_id', mysql_cn).values.tolist()]
    artist_set = set(res_list)
    
    dic = {}
    res_list = pd.read_sql('''
    SELECT songs.artist_id, user_actions.ds
    FROM user_actions JOIN songs
    on user_actions.song_id = songs.song_id
    WHERE user_actions.action_type = 1 and (
    user_actions.ds = '20150519' or 
    user_actions.ds = '20150522' or 
    user_actions.ds = '20150725' or 
    user_actions.ds = '20150731')
    GROUP BY artist_id, ds
    order by ds asc
    ''', mysql_cn).values.tolist()
    for item in res_list:
        if dic.has_key(item[1]):
            dic[item[1]].append(item[0])
        else:
            dic[item[1]] = [item[0]]
            
    for key in dic.keys():
        print '%s miss %s'%(key, artist_set - set(dic[key]))
        
    mysql_cn.close()

def calc_artist_strange():
    mysql_cn= pymysql.connect(host='10.41.0.63', port=3306,user='root', passwd='111111', db='music')
    strange_artist_list = ['2b7fedeea967becd9408b896de8ff903', '4ee3f9c90101073c99d5440b41f07daa',
                           'bf21d16799b240d6e445fa30472bd50b']
    sql_template = '''
    SELECT count(*) as plays, ds from 
    user_actions JOIN(
    SELECT * from songs WHERE artist_id = '{artist_id}')a
    on user_actions.song_id = a.song_id
    GROUP BY ds;
    '''
    for artist_id in strange_artist_list:
        artist_strange = pd.read_sql(sql_template.format(artist_id = artist_id), mysql_cn)
        ds = artist_strange.pop('ds').values.tolist()
        artist_strange.plot()
        artist_strange.index = ds
        print artist_strange
        plt.show()
    mysql_cn.close()
    
if __name__ == '__main__':
#     calc_artist_miss()
    calc_artist_strange()