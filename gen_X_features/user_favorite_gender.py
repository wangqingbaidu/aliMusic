# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 26, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''

featueName = 'favorite_gender'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit

def gen(fromDate = None, toDate = None, table_name = None):
    
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    select b.user_id,
        max(if(b.gender='1', b.fav_gender_times,NULL) )as gender1,
        max(if(b.gender='2', b.fav_gender_times,NULL) )as gender2,
        max(if(b.gender='3', b.fav_gender_times,NULL) )as gender3
    FROM(
    SELECT user_id, gender, SUM(times) as fav_gender_times from(
    SELECT user_id, user_actions.song_id, songs.artist_id, gender, count(*) as times FROM
    user_actions join songs
    on user_actions.song_id = songs.song_id
    where user_actions.ds >= '{fromDate}' and user_actions.ds <= '{toDate}'
    GROUP BY user_id, user_actions.song_id)a
    GROUP BY user_id, gender
    ORDER BY user_id, fav_gender_times desc
    )b
    GROUP BY b.user_id;
    create index IDX_{table_name} on {table_name}(user_id);
 """
    return sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate)
    

def genAll():
    return (gen(fromDateTrain, toDateTrain, 'user_%s_train' %featueName), 
            gen(fromDateTest, toDateTest, 'user_%s_test' %featueName),
            gen(fromDateSubmit, toDateSubmit, 'user_%s_submit' %featueName))
    
if __name__ == '__main__':
    for sql in genAll():
        print sql