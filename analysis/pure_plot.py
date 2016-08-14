# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jul 7, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import pymysql
import matplotlib                                                                        
from datetime import datetime
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_artist():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='alimusic')
    artist_list = [(x[0], x[1]) 
                      for x in pd.read_sql('select artist_id, plays from artist_list order by plays desc', 
                                           mysql_cn).values.tolist()]
    count = 0
    for artist_id, _ in artist_list:
        df = pd.read_sql('''
        SELECT plays from artist_play
        WHERE artist_id = '{artist_id}'
        order by ds;
        '''.format(artist_id=artist_id), mysql_cn)
        df2 = pd.read_sql('''
        SELECT plays from clean_artist_play_new_songs_out
        WHERE artist_id = '{artist_id}'
        order by ds;
        '''.format(artist_id=artist_id), mysql_cn)
        df['out'] = df2['plays']
        df.columns = ['origin', 'song_out']
        df.plot()
        plt.title('%s' %(artist_id))
        print artist_id, 'Got!'
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        fig.savefig('./p2_img/p2_conbine_{No:0>3}_{artist_id}.png'.format(artist_id = artist_id,
                                                                  No = count))
        plt.close(fig)
        count += 1
    mysql_cn.close()
    
def plot_album():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='music')
    album_list = [(x[0], x[1]) 
                      for x in pd.read_sql('select album, plays from sept_album_list order by album desc', 
                                           mysql_cn).values.tolist()]
    
    for album, _ in album_list:
        df = pd.read_sql('''
        SELECT plays from sept_new_songs_plays
        WHERE album = '{album}'
        order by ds;
        '''.format(album=album), mysql_cn)
        df.plot()
        plt.title('%s' %(album))
        fig = plt.gcf()
        fig.savefig('./img/sept_{album}.png'.format(album = album))
        plt.close(fig)
    mysql_cn.close()
    
def plot_ahead_songs():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='alimusic')
    get_ahead_album = '''
    SELECT album FROM new_songs_plays GROUP BY album
    HAVING  SUBSTRING(album, CHAR_LENGTH(album) - 7, CHAR_LENGTH(album) ) > MIN(ds);
    '''
    album_list = [x[0] for x in pd.read_sql(get_ahead_album, mysql_cn).values.tolist()]
    for album in album_list:
#         if album[-8:] < '20150830':
#             continue
        df = pd.read_sql('''
        SELECT plays, ds from new_songs_plays
        WHERE album = '{album}'
        order by ds;
        '''.format(album = album), mysql_cn)
        start = df.iloc[0:1, 1:2].values[0][0]
        print start, album[-8:], album
        x = (datetime.strptime(album[-8:],'%Y%m%d') - datetime.strptime(start,'%Y%m%d')).days
        df['plays'].plot()
        plt.title('%s' %(album))
        print x, df['plays'].values.shape
        y = df.ix[df.ds==album[-8:],"plays"].values.tolist()
        if not y:
            y = 0
        plt.plot(x, y, 'o', color='red')
        
        fig = plt.gcf()
        fig.savefig('./p2_img/ahead_{album}.png'.format(album = album))
        plt.close(fig)
def plot_predict_ahead_songs():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='alimusic')
    album_list = [x[0] for x in pd.read_sql('select album from sept_album_list', mysql_cn).values.tolist()]
    for album in album_list:
        df = pd.read_sql('''
        SELECT plays from sept_new_songs_plays
        WHERE album = '{album}'
        order by ds;
        '''.format(album = album), mysql_cn)
        df.plot()
        plt.title('%s' %(album))
        fig = plt.gcf()
        fig.savefig('./p2_img/sept_ahead_{album}.png'.format(album = album))
        plt.close(fig)
        print album, 'got!'
        
def d9513():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='alimusic')
    df1 = pd.read_sql('''
    SELECT a.plays as pbefore, b.plays as pafter from clean_artist_play a
    LEFT JOIN d9513 b
    ON a.ds = b.ds
    WHERE a.artist_id like 'd9513%'
    ''', mysql_cn)    

    df1.plot()        
    plt.show()
        
        
def plot_all_artist():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='alimusic')
    artist_list = [(x[0], x[1]) 
                      for x in pd.read_sql('select artist_id, plays from artist_list order by plays desc', 
                                           mysql_cn).values.tolist()]
    count = 0
    for artist_id, _ in artist_list:
        df = pd.read_sql('''
        SELECT plays from final_combine_all_artist_plays
        WHERE artist_id like '{artist_id}'
        order by ds;
        '''.format(artist_id=artist_id), mysql_cn)
        df.plot()
        plt.title('%s' %(artist_id))
        print artist_id, 'Got!'
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        fig.savefig('./all_artist_plays/p2_conbine_{No:0>3}_final_{artist_id}.png'.format(artist_id = artist_id,
                                                                  No = count))
        plt.close(fig)
        count += 1
    mysql_cn.close()
if __name__ == '__main__':
#     plot_predict_ahead_songs()
#     d9513()
    plot_all_artist()
#     zzw = []
#     for f in os.listdir('C:\Users\wangqingbaidu\Desktop\img'):
#         zzw.append(f.split('_')[-1])
#     zg = []   
#     for i in open('C:\Users\wangqingbaidu\Desktop\artist.txt').readlines():
#         zg.append(i)
        
    