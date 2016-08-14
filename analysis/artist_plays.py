# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jun 6, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt

def analysis():
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='music')
    sqlTemplate = '''
    SELECT COUNT(*) as plays from user_actions left JOIN songs
    on user_actions.song_id = songs.song_id
    where ds >= '20150301' and ds <= '20150830' AND songs.artist_id = 'b15e8846dc61824c1242a6b36796117b' 
    and action_type = '1'
    GROUP BY ds
    order by ds;
    '''
    sql = '''
    SELECT plays from artist_play
    WHERE artist_id like 'd13%'
    '''
    df = pd.read_sql(sql, mysql_cn)
#     df['plays'] = df['plays'] / 6000
    df.plot()
    plt.show()
#     fig = plt.gcf()
#     fig.savefig('c5f0170f87a2fbb17bf65dc858c745e2_plays.png')
    
    mysql_cn.close()
    
    
if __name__ == '__main__':
    analysis()
