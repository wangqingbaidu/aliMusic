# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 25, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''

featueName = 'artist_taste'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit
from gen_X_features import totalArtists

def gen(fromDate = None, toDate = None, table_name = None, action_type = None):
    if not action_type or len(action_type) == 0:
        action_type = [1,2,3]
        
    action_condition = ' '.join(['or user_actions.action_type={action_type}'.format(action_type=i)
                        for i in action_type])
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    select c.user_id,
    {taste_matrix}
    from(
    SELECT user_id, label, times from(
    SELECT  user_actions.user_id, songs.artist_id, count(*) as times
    FROM 
    user_actions join songs
    on user_actions.song_id = songs.song_id
    where user_actions.ds >= '{fromDate}' and user_actions.ds <= '{toDate}'
    {action_condition}
    GROUP BY user_actions.user_id, songs.artist_id
    ORDER BY  user_actions.user_id, times desc, songs.artist_id  desc) a
    JOIN artist_label_map
    on a.artist_id = artist_label_map.artist_id)c
    GROUP BY user_id;
 """
    taste_matrix = ",\n\t".join(['max(if(c.label={index},1,NULL) )as has_a{index}'.format(index = index + 1) 
                                 for index in range(totalArtists)])
    return sqlTemplate.format(action_condition = action_condition, taste_matrix = taste_matrix,
                              table_name = table_name, fromDate = fromDate, toDate = toDate)
    

def genAll():
    return (gen(fromDateTrain, toDateTrain, 'user_%s_train' %featueName), 
            gen(fromDateTest, toDateTest, 'user_%s_test' %featueName),
            gen(fromDateSubmit, toDateSubmit, 'user_%s_submit' %featueName))
    
if __name__ == '__main__':
    for sql in genAll():
        print sql