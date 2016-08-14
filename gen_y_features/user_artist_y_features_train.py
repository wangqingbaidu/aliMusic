# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 13, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import datetime
from gen_y_features import fromDateTrain as fromDate, toDateTrain as toDate
def gen(table_name = 'y_train_user', ignore_date = True):
    assert table_name
    sql = ''
    if not ignore_date:
        sqlTemplate = """
        drop table if exists {table_name};
        create table {table_name} as
        select user_actions.user_id, songs.artist_id, count(*) as play_times, user_actions.ds
        from 
        user_actions LEFT JOIN songs on user_actions.song_id = songs.song_id
        where user_actions.ds >='20150501' and user_actions.ds <= '20150630' and user_actions.action_type='1'
--        and songs.artist_id <> '2b7fedeea967becd9408b896de8ff903'
        group by user_id, artist_id, ds;
        create index IDX_{table_name} on {table_name}(user_id);
        """
        sql = sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate)
    else:
        sqlTemplate = """
        drop table if exists {table_name};
        create table {table_name} as
        select user_actions.user_id, songs.artist_id, count(*) / {total_days} as play_times
        from 
        user_actions LEFT JOIN songs on user_actions.song_id = songs.song_id
        where user_actions.ds >='20150501' and user_actions.ds <= '20150630' and user_actions.action_type='1'
--        and songs.artist_id <> '2b7fedeea967becd9408b896de8ff903'
        group by user_id, artist_id;
        create index IDX_{table_name} on {table_name}(user_id);
        """
        total_days = (datetime.datetime.strptime(toDate, "%Y%m%d").date() - 
                      datetime.datetime.strptime(fromDate, "%Y%m%d").date()).total_seconds() / (24 * 60 * 60) + 1
        sql = sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate,
                                 total_days = total_days)
        
    return sql

# def gen_ignore_date(table_name = None):
#     assert table_name
#     sqlTemplate = """
#     drop table if exists {table_name};
#     create table {table_name} as
#     select user_actions.user_id, songs.artist_id, count(*) as play_times
#     from 
#     user_actions LEFT JOIN songs on user_actions.song_id = songs.song_id
#     where user_actions.ds >='20150501' and user_actions.ds <= '20150630' and user_actions.action_type='1'
#     group by user_id, artist_id;
#     create index IDX_{table_name} on {table_name}(user_id);
#     """
#     sql = sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate)
#     
#     return sql
if __name__ == '__main__':
    print gen('y_train_user', True)