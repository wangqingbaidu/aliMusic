# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 24, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''

featueName = 'favorite_artist'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit
from gen_X_features import totalArtists
def gen(fromDate = None, toDate = None, table_name = None):
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    select c.user_id,
    {artist_matrix}
    from(
    SELECT user_id, label, times FROM
    (SELECT  user_actions.user_id, songs.artist_id, count(*) as times
    FROM user_actions join songs
    on user_actions.song_id = songs.song_id
    where user_actions.ds >= '{fromDate}' and user_actions.ds <= '{toDate}'
    GROUP BY user_actions.user_id, songs.artist_id
    ORDER BY  user_actions.user_id, times desc, songs.artist_id  desc)a JOIN artist_label_map
    on a.artist_id = artist_label_map.artist_id
    )c GROUP BY c.user_id;
    create index IDX_{table_name} on {table_name}(user_id);
 """
    artist_matrix = ",\n\t".join(['max(if(c.label={index},times,NULL) )as fav_artist{index}'.format(index = index + 1) 
                                 for index in range(totalArtists)])
    return sqlTemplate.format(table_name=table_name, fromDate=fromDate, toDate=toDate,
                              artist_matrix = artist_matrix)
    

def genAll():
    return (gen(fromDateTrain, toDateTrain, 'user_%s_train' %featueName), 
            gen(fromDateTest, toDateTest, 'user_%s_test' %featueName),
            gen(fromDateSubmit, toDateSubmit, 'user_%s_submit' %featueName))
    
if __name__ == '__main__':
    for sql in genAll():
        print sql