# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jun 2, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import datetime

ratio_days = [1, 3, 5, 7, 15, 30]
featueName = 'plays_ratio'
from gen_X_features import toDateTrain, toDateTest, toDateSubmit

def gen(toDate = None, table_name = None, sub_day = None):
    assert sub_day
    gapday = datetime.timedelta(days=sub_day - 1)
    fromDate_A = datetime.datetime.strptime(toDate, "%Y%m%d").date() - gapday
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    SELECT user_actions.user_id, songs.artist_id, count(*) as plays FROM 
    user_actions JOIN songs
    ON user_actions.song_id = songs.song_id
    WHERE user_actions.action_type = 1 AND
    ds >= '{fromDate}' AND ds <= '{toDate}'
    GROUP BY user_actions.user_id, songs.artist_id
    ORDER BY user_actions.user_id;
    create index IDX_{table_name} on {table_name}(user_id, artist_id);
 """
    return sqlTemplate.format(table_name = table_name, 
                              fromDate = fromDate_A.strftime('%Y%m%d'), 
                              toDate = toDate)
    

def genAll():
    res = []
    for day in ratio_days:
        res += [gen(toDateTrain, 'user_%s_%d_train' %(featueName, day), day), 
            gen(toDateTest, 'user_%s_%d_test' %(featueName, day), day),
            gen(toDateSubmit, 'user_%s_%d_submit'%(featueName, day), day)]
    return res
    
if __name__ == '__main__':
    for sql in genAll():
        print sql