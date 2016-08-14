# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 26, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import datetime

featueName = 'favorite_language'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit

def gen(fromDate = None, toDate = None, table_name = None):
    language_set = [0, 1, 2, 3, 4, 11, 12, 14, 100]
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    select b.user_id,
    {language_matrix}
    FROM(
    SELECT user_id, language, SUM(times) as fav_language_times from(
    SELECT user_id, user_actions.song_id, songs.artist_id, LANGUAGE, count(*) as times FROM
    user_actions join songs
    on user_actions.song_id = songs.song_id
    where user_actions.ds >= '{fromDate}' and user_actions.ds <= '{toDate}'
    GROUP BY user_id, user_actions.song_id)a
    GROUP BY user_id, language
    ORDER BY user_id, fav_language_times desc
    )b
    GROUP BY b.user_id;
    create index IDX_{table_name} on {table_name}(user_id);
 """
    language_matrix = ",\n\t".join(["max(if(b.language='{l}',fav_language_times,NULL) )as fav_lan{l}".format(l = l) 
                                 for l in language_set])
    return sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate,
                              language_matrix = language_matrix)
    

def genAll():
    return (gen(fromDateTrain, toDateTrain, 'user_%s_train' %featueName), 
            gen(fromDateTest, toDateTest, 'user_%s_test' %featueName),
            gen(fromDateSubmit, toDateSubmit, 'user_%s_submit' %featueName))
    
if __name__ == '__main__':
    for sql in genAll():
        print sql