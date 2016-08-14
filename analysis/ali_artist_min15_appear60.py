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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

def get_filtered_artist(dateSplit = None):
    mysql_cn= pymysql.connect(host='10.25.0.118', port=3306,user='root', passwd='111111', db='music')
    artist_list = pd.read_sql('select artist_id, plays from artist_list order by plays desc', 
                              mysql_cn).values.tolist()
    artist_list = [(x[0], x[1]) for x in artist_list]
    count = 0.0
    count_appear = 0
    ratio = 0
    a_count = 0
    x = datetime.datetime.strptime(dateSplit, "%Y%m%d").date() - \
    datetime.datetime.strptime('20150301', "%Y%m%d").date()
    filter_artist_list = []
    for artist, plays in artist_list:
        df = pd.read_sql('''
        SELECT plays from artist_play
        WHERE artist_id = '{artist}'
        order by ds;
        '''.format(artist=artist),mysql_cn)
#         if 'd13' in artist:
#             print abs(df.values - np.mean(df.values)) * 1.0 / np.std(df.values)
#         print x.days        
        if np.max(abs(df.values[- 30: ] - np.mean(df.values[-30:])) * 1.0 / 
                  np.std(df.values[-30: ])) <= 3:
#         if np.max(abs(df.values[x.days - 30: x.days] - np.mean(df.values[x.days-30: x.days])) * 1.0 / 
#                   np.std(df.values[x.days-30: x.days])) <= 3:
            continue
        a_count+=1
#         print ''' \t\tor artist_id = '%s' ''' %artist
        filter_artist_list.append(artist)
        for i in range(15, 124):
#             if np.min(df[i - 15 : i].values) >= np.min(df[i : i+60].values) * (1 - ratio) and \
#                 np.min(df[i - 15 : i].values) <= np.min(df[i : i+60].values) * (1 + ratio):
            if np.min(df[i - 15 : i].values) >= np.min(df[i : i+60].values):
                count_appear += 1
            count += 1
#     print "Total artist:{a_count}. Ratio:{ratio}".format(a_count = a_count, ratio = count_appear / count)
    mysql_cn.close()
    return (filter_artist_list, count_appear / count)

def gen_sql(dateSplit= None, use_days = 15, having_counts = None, predict_days = 60, use_clean_data=True, use_artist = True):
    assert dateSplit
    if not having_counts:
        having_counts = use_days
    pgapday = datetime.timedelta(days=use_days - 1)
    tfgapday = datetime.timedelta(days=2)
    ttgapday = datetime.timedelta(days=predict_days + 1)
    pfromDate = datetime.datetime.strptime(dateSplit, "%Y%m%d").date() - pgapday
    tfromDate = datetime.datetime.strptime(dateSplit, "%Y%m%d").date() + tfgapday
    ttoDate = datetime.datetime.strptime(dateSplit, "%Y%m%d").date() + ttgapday
    artist_and_condition = ''
    artist_or_condition = []
    if use_artist:
        artist_list, prob = get_filtered_artist(dateSplit)
        for artist in artist_list:
            artist_and_condition += '''\t\t\t\tand artist_id != '{artist}' \n'''.format(artist=artist)
            artist_or_condition.append("artist_id = '{artist}'\n".format(artist=artist))
    data2use = {True:'clean_user_actions_with_artists', False:'user_actions_with_artists'}
    local_test = '''
    select sum(f.fscore) as fscore_best_1, '1' as orders from(
      select  artist_id,sqrt(sum(e.target)) * (1-sqrt(avg(pow((e.plays-e.target)/e.target,2)))) as fscore
      from(
        select c.artist_id,c.ds,c.target,d.plays from(
            select t.artist_id,t.ds,t.target
            from (
                select artist_id, ds, count(1) as target from
                user_actions_with_artists
                where action_type = 1
                and ds >= '{tfromDate}'
                and ds <= '{ttoDate}' 
                group by artist_id, ds
            )t
        )c
        left outer join
        (
          select artist_id,sum(1/plays)/sum(1/plays/plays) as plays
          from(
              select artist_id,ds,count(1) as plays
              from {data2use}
              where action_type=1                
                 and ds >= "{pfromDate}"
                 and ds <= "{ptoDate}"    
              group by artist_id,ds
          )b group by artist_id
    
        )d on(c.artist_id=d.artist_id)
      )e group  by artist_id
    )f
    '''.format(tfromDate = tfromDate.strftime('%Y%m%d'), 
               ttoDate = ttoDate.strftime('%Y%m%d'), 
               pfromDate = pfromDate.strftime('%Y%m%d'), 
               ptoDate = dateSplit,
               data2use = data2use[use_clean_data])
    artist_pred = '''
    select tmp.* from
        (
          select b.artist_id,sum(1/b.plays)/sum(1/b.plays/b.plays) as plays
          from(
              select artist_id,ds,count(1) as plays
              from {data2use}
              where action_type=1
              and ds >= "{pfromDate}"
              and ds <= "{ptoDate}"
        {artist_and_condition}
              group by artist_id,ds
          )b group by b.artist_id
        union all 
--        select a.artist_id, cast(min(a.plays) as double) as plays 
        select a.artist_id, sum(1/a.plays)/sum(1/a.plays/a.plays) * {prob} as plays
        from(
          select artist_id,ds,count(1) as plays from {data2use}
          where action_type=1
             and ds >= "{pfromDate}"
             and ds <= "{ptoDate}"        
             and ( {artist_or_condition}\t\t)
          group by artist_id,ds
        )a group by a.artist_id
      )tmp
    '''
    
    song_pred = '''
    select ar.artist_id, sum(so.plays) as plays from
          songs ar
        join(
          select song_id,sum(1/plays)/sum(1/plays/plays) as plays from (
            select s.song_id, b.ds, b.plays from(
              select a.song_id from( 
                select song_id, ds, count(1) as plays from {data2use} 
                where action_type = 1
                and ds >= "{pfromDate}"
                and ds <= "{ptoDate}" 
                group by song_id, ds
                )a
              group by a.song_id having count(1) >= {having_counts}
            )s
            join(
                select song_id, ds, count(1) as plays from {data2use} 
                where action_type = 1
                and ds >= "{pfromDate}"
                and ds <= "{ptoDate}" 
                group by song_id, ds
            )b
            on b.song_id = s.song_id 
          )x group by song_id
        )so
        on so.song_id = ar.song_id
        group by ar.artist_id
    '''
    pred = artist_pred.format(pfromDate = pfromDate.strftime('%Y%m%d'), 
                              ptoDate = dateSplit,
                              artist_and_condition = artist_and_condition,
                              artist_or_condition = '\t\t\t\tor '.join(artist_or_condition),
                              data2use = data2use[use_clean_data],
                              prob = 1.1)
    if not use_artist:
        pred = song_pred.format(pfromDate = pfromDate.strftime('%Y%m%d'), 
                                ptoDate = dateSplit,
                                data2use = data2use[use_clean_data],
                                having_counts = having_counts)
    target = '''
    select t.artist_id,t.ds,t.target
        from (
            select artist_id, ds, count(1) as target from
            user_actions_with_artists
            where action_type = 1
            and ds >= '{tfromDate}'
            and ds <= '{ttoDate}'
            group by artist_id, ds
        )t
    '''
    
    test = '''
select sum(f.fscore) as fscore_best_1, '2' as orders from(
  select  artist_id,sqrt(sum(e.target)) * (1-sqrt(avg(pow((e.plays-e.target)/e.target,2)))) as fscore
  from(
    select c.artist_id,c.ds,c.target,d.plays from(
        {target}
    )c
    left outer join
    (
        {pred}
    )d on(c.artist_id=d.artist_id)
  )e group  by artist_id
)f
    '''.format(target = target.format(tfromDate = tfromDate.strftime('%Y%m%d'), 
                                              ttoDate = ttoDate.strftime('%Y%m%d')),
                       pred = pred)
    res = '''select * from 
({local_test}
union all
{test})tmp
order by orders limit 2
    '''.format(local_test = local_test, test = test)
    
    return res
if __name__ == '__main__':
    print gen_sql('20150630', use_days = 14, having_counts = None, use_clean_data=True,  use_artist= False)
