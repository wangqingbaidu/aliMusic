# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 13, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
from gen_y_features import fromDateTrain as fromDate, toDateTrain as toDate
def gen(table_name = 'y_train_user', ignore_date = True):
    assert table_name
    sql = ''
    if not ignore_date:
        sqlTemplate = """
        drop table if exists {table_name};
        create table {table_name} as
        select user_id, song_id, count(*) as play_times, ds
        from user_actions 
        where ds >='{fromDate}' and ds <= '{toDate}' and action_type='1'
        group by user_id, song_id, ds;
        create index IDX_{table_name} on {table_name}(user_id);
        """
        sql = sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate)
    else:
        sqlTemplate = """
        drop table if exists {table_name};
        create table {table_name} as
        select user_id, song_id, count(*) as play_times
        from user_actions 
        where ds >='{fromDate}' and ds <= '{toDate}' and action_type='1'
        group by user_id, song_id;
        create index IDX_{table_name} on {table_name}(user_id);
        """
        sql = sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate)     
    
    return sql

# def gen_ignore_date(table_name = None):
#     assert table_name
#     sqlTemplate = """
#     drop table if exists {table_name};
#     create table {table_name} as
#     select user_id, song_id, count(*) as play_times
#     from user_actions 
#     where ds >='{fromDate}' and ds <= '{toDate}' and action_type='1'
#     group by user_id, song_id;
#     create index IDX_{table_name} on {table_name}(user_id);
#     """
#     sql = sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate)
#     
#     return sql
if __name__ == '__main__':
    print gen('y_train_user')