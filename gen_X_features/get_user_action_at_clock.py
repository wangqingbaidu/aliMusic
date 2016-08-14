# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 10, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
def gen():
    sql = ''
    sqlTemplate = """
CREATE TABLE user_play_at{0} AS 
SELECT user_id, total from tmp_user_action_distinguish_by_clock
 where action_type='1' and action_time='{0:0>2}';
CREATE TABLE user_download_at{0} AS SELECT user_id, total from tmp_user_action_distinguish_by_clock where action_type='2' and action_time='{0:0>2}';
CREATE TABLE user_collect_at{0} AS SELECT user_id, total from tmp_user_action_distinguish_by_clock where action_type='3' and action_time='{0:0>2}';
    """
    for i in range(24):
        sql += sqlTemplate.format(i)
    
    return sql
if __name__ == '__main__':
    print gen()