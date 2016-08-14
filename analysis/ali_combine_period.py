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

def gen_sql(dateSplit= None, use_days = 15,  predict_days = 60, use_clean_data=True):
    assert dateSplit
    partition_p_fromDate = ['20150902', '20150922', '20151011']
    partition_p_toDate = ['20150921', '20151010', '20151030']
    partition_u_fromDate = ['20150824', '20150817', '20150810']
    partition_u_toDate = ['20150830', '20150830', '20150830']
    pgapday = datetime.timedelta(days=use_days - 1)
    tfgapday = datetime.timedelta(days=2)
    ttgapday = datetime.timedelta(days=predict_days + 1)
    pfromDate = datetime.datetime.strptime(dateSplit, "%Y%m%d").date() - pgapday
    tfromDate = datetime.datetime.strptime(dateSplit, "%Y%m%d").date() + tfgapday
    ttoDate = datetime.datetime.strptime(dateSplit, "%Y%m%d").date() + ttgapday
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
    
    partition_pred = '''
    select k.*, cast(cast({tname}.plays+0.5 as int) as string) as plays from (
      select artist_id, ds from total_keys
      where ds >= "{pfromDate}"
        and ds <= "{ptoDate}"
      ) k
      left outer join(
        select artist_id,sum(1/plays)/sum(1/plays/plays) as plays
        from(
            select artist_id,ds,count(1) as plays
            from {data2use}
            where action_type=1                
               and ds >= "{ufromDate}"
               and ds <= "{utoDate}"
            group by artist_id,ds
        )b group by artist_id
      ){tname}
    on k.artist_id = {tname}.artist_id
    '''
    pred = 'union all'.join([partition_pred.format(tname = 't%s_%s' %(ppfromDate, pptoDate),
                                                   pfromDate = ppfromDate,
                                                   ptoDate = pptoDate,
                                                   ufromDate = pufromDate,
                                                   utoDate = putoDate,
                                                   data2use = data2use[use_clean_data])
                            for ppfromDate, pptoDate, pufromDate, putoDate in zip(partition_p_fromDate,
                                                                                  partition_p_toDate,
                                                                                  partition_u_fromDate,
                                                                                  partition_u_toDate)])
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
    )d on(c.artist_id=d.artist_id and c.ds = d.ds)
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
    print gen_sql('20150830', use_days = 14, use_clean_data=True)
