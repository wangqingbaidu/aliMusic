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

def artist_plays():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='music')
    
#     df = pd.read_sql("""
#     SELECT * from (    
#     SELECT test_3000_lines.artist_id, plays, test_3000_lines.ds from 
#         test_3000_lines LEFT JOIN(
#         SELECT artist_id, count(*) as plays, ds from
#         user_actions left JOIN songs
#         on user_actions.song_id = songs.song_id
#         WHERE ds >= '20150702' and ds <= '20150830' and action_type = '1'
#         GROUP BY ds, artist_id
#         ORDER BY artist_id, ds)a
#         on test_3000_lines.artist_id = a.artist_id and test_3000_lines.ds = a.ds
#         ORDER BY ds
#         LIMIT 50, 3000)a
#     ORDER BY artist_id, ds
#     """, mysql_cn)
    artist_list = pd.read_sql('select artist_id, plays from artist_list order by artist_id desc', 
                              mysql_cn).values.tolist()
    artist_list = [(x[0], x[1]) for x in artist_list]
    count = 0
    for artist, plays in artist_list:
#         if plays < 60000:
#             continue
#         if artist != '6d4ecd79fd8a64039ecbfcce36019218':
#             continue
        df = pd.read_sql('''
        SELECT plays from  clean_user_artist_by_ds_new_song_out
        WHERE artist_id = '{artist}'
        order by ds;
        '''.format(artist=artist),mysql_cn)
        df2 = pd.read_sql('''
        SELECT plays from artist_play
        WHERE artist_id = '{artist}'
        order by ds;
        '''.format(artist=artist),mysql_cn)
        df3 = pd.read_sql('''
        SELECT plays from clean_user_artist_by_ds
        WHERE artist_id = '{artist}'
        order by ds;
        '''.format(artist=artist),mysql_cn)
        df['dirty'] = df3['plays']
        df['clean'] = df3['plays'] - df['plays']
        df = df.astype(int)
        df.columns = ['without','dirty', 'residual']
        df.plot()
#         df['without'].plot()
#         plt.show()
        fig = plt.gcf()
        fig.savefig('./img/new_song_residual_No{No:0>3}_{artist}.png'.format(artist=artist, No = count))
        count += 1
        print 'Artist {artist} got!'.format(artist=artist)
    mysql_cn.close()
    
def artist_decay_plays():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='music')
    artist_list = pd.read_sql('select artist_id, plays from artist_list order by artist_id desc', 
                              mysql_cn).values.tolist()
    artist_list = [(x[0], x[1]) for x in artist_list]
    count = 0
    artist_string = '''
1731019fbaa825714d5f8e61ad1bb7ff
300dde62b988fc83eb6dba1b702e9af3
509735a54607273699d6fb9496daccf4
5dbd0882d237a4830436a7f7986e59ab
75808a38c82695330edc632d562a0238
b26955262273f5cc9a276f792e51efe0
e7912240b9fb5bb5f560a8855f626789
efbbfd2611307e97bd36298b2fce2a02
f67f8584e121c09bc6f514ed02b876df
    '''
    for artist, plays in artist_list:
        if not artist in artist_string:
            continue
        df = pd.read_sql('''
        SELECT plays from  test_incr
        WHERE artist_id = '{artist}'
        order by ds;
        '''.format(artist=artist),mysql_cn)
        df2 = pd.read_sql('''
        SELECT plays from artist_play
        WHERE artist_id = '{artist}'
        and ds >= '20150702' and ds <= '20150830'
        order by ds;
        '''.format(artist=artist),mysql_cn)
        df['origin'] = df2['plays']
        df = df.astype(float)
        df.columns = ['decay','origin']
        df.plot()
#         df['without'].plot()
#         plt.show()
        fig = plt.gcf()
        fig.savefig('./img/incr_No{No:0>3}_{artist}.png'.format(artist=artist, No = count))
        count += 1
        print 'Artist {artist} got!'.format(artist=artist)
    mysql_cn.close()
    
if __name__ == '__main__':
    artist_decay_plays()
#     artist_plays()

'''
47c05597b30c1fc870d2dba43e318fdb        
        
        
'''