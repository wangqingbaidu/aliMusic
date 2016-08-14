# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jun 19, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import pymysql
import matplotlib                                                                        
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='music')
def get_min_error_res(play_res):
    res_sum = 0
    res_sum_2 = 0
    for res in play_res:
        if res < 1: continue
        res_sum += 1.0/res
        res_sum_2 += 1.0/(res*res)
    if res_sum == 0: return 0
    return res_sum / res_sum_2

def get_min_error_mean_results(queryType = None):
    """
    in_filename: artist_id, times, ds
    """
    assert queryType
    keys = ['artist_id', 'times', 'ds']
    artist = {}
#     data = pd.read_csv(in_filename, header = None, names =  keys)
    data = None
    if queryType == 16:
        data = pd.read_sql('''
        select artist_id, plays, ds from artist_play where ds >= '20150815' and ds <= '20150830' 
        ''', mysql_cn)
    elif queryType == 15:
        data = pd.read_sql('''
        select artist_id, plays, ds from artist_play where ds >= '20150816' and ds <= '20150830' 
        ''', mysql_cn)
    elif queryType == 14:
        data = pd.read_sql('''
        select artist_id, plays, ds from artist_play where ds >= '20150817' and ds <= '20150830' 
        ''', mysql_cn)
    else:
        assert False
    
    data.columns = keys
    days = set()
    for _, row in data.iterrows():
        artist_id = row[keys[0]]
        if artist_id not in artist:
            artist[artist_id] = []
        artist[artist_id].append(row[keys[1]])
        days.add(row[keys[2]])
    days = [day for day in days]
    sorted(days)
    results = []
    for artist_id, times in artist.iteritems():
        min_error_res = int(get_min_error_res(times))
        for day in days:
            results.append([artist_id, min_error_res, day])
    df = pd.DataFrame(results)
    df.columns = ['artist_id', 'plays', 'ds']
    df.pop('ds')
    df = df.groupby(by='artist_id')['plays'].mean().reset_index()
    df.columns = ['artist_id', 'plays']
    df = df.sort_values(by = 'artist_id')
    return df
    
def analysis():
    avg_14 = pd.read_sql('''
    SELECT artist_id, avg(plays) as plays FROM artist_play
    WHERE ds >= '20150817' AND ds <= '20150830'
    GROUP BY artist_id
    order by artist_id;
    ''', mysql_cn)    
    avg_15 = pd.read_sql('''
    SELECT artist_id, avg(plays) as plays FROM artist_play
    WHERE ds >= '20150816' AND ds <= '20150830'
    GROUP BY artist_id
    order by artist_id;
    ''', mysql_cn)      
    avg_16 = pd.read_sql('''
    SELECT artist_id, avg(plays) as plays FROM artist_play
    WHERE ds >= '20150815' AND ds <= '20150830'
    GROUP BY artist_id
    order by artist_id;
    ''', mysql_cn)    
    
    print avg_14.iloc[[74,78]]
    me_14 = get_min_error_mean_results(14)
    me_15 = get_min_error_mean_results(15)
    me_16 = get_min_error_mean_results(16)
    
    dropIndex = [78, 74]
    avg_14 = avg_14.drop(dropIndex)
    me_14 = me_14.drop(dropIndex)    
    avg_15 = avg_15.drop(dropIndex)
    me_15 = me_15.drop(dropIndex)    
    avg_16 = avg_16.drop(dropIndex)
    me_16 = me_16.drop(dropIndex)
    avg = [avg_14, avg_15, avg_16]
    me = [me_14, me_15, me_16]
    s = [avg_15, me_14]
    s = [me_14['plays']/ x['plays'] for x in s]
    x = pd.DataFrame(me_14['plays']/ avg_15['plays'])
    print x
    x.columns = ['plays']
    x = x.sort_values(by = 'plays')
    
    print x.iloc[49]
    df_show = pd.concat(s, axis = 1, ignore_index=True)
    
    df_show.columns = ['me_14',
#                        'me_15',
#                        'me_16',
                        'avg_14',  
#                        'avg_15',  
#                        'avg_16'
                       ]    
#     df_show.columns = ['avg_14', 'me_14']
    
    df_show.plot()
    plt.show()
#     fig = plt.gcf()
#     fig.savefig('./img/compare_plays.png')
    mysql_cn.close()
    
    
if __name__ == '__main__':
    analysis()
