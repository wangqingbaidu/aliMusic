# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 15, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''

featueName = 'top_5period_play'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit

def gen_tmp_table(fromDate = None, toDate = None, table_name = None):
    sqlTemplate = """
    CREATE TABLE if not exists tmp_{table_name} AS 
    SELECT user_id, FROM_UNIXTIME( cast(gmt_create AS UNSIGNED), '%H' ) AS hour_of_day, count(*) AS counts 
    FROM user_actions 
    where action_type = '1' and ds >= '{fromDate}' and ds <= '{toDate}'
    GROUP BY user_id, action_type, hour_of_day;
    """
    return sqlTemplate.format(table_name=table_name, toDate=toDate, fromDate=fromDate)

def gen(fromDate = None, toDate = None, table_name = None):
    fromTableName = None
    if 'train' in table_name:
        fromTableName = 'tmp_top_5train'
    elif 'test' in table_name:
        fromTableName = 'tmp_top_5test'
    elif 'submit' in table_name:
        fromTableName = 'tmp_top_5submit'
    else:
        assert False
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    select c.user_id,
        max(if(c.rank=1,c.hour_of_day,-1)) as rank1,
        max(if(c.rank=2,c.hour_of_day,-1)) as rank2,
        max(if(c.rank=3,c.hour_of_day,-1)) as rank3,
        max(if(c.rank=4,c.hour_of_day,-1)) as rank4,
        max(if(c.rank=5,c.hour_of_day,-1)) as rank5
        from (
        select d.user_id,d.hour_of_day,d.counts,
        @rownum:=@rownum+1,
        if(@pu=d.user_id,@rank:=@rank+1,@rank:=1) as rank,@pu:=d.user_id
        from(
            select user_id,hour_of_day,counts 
            from {fromTableName}
            order by user_id,counts DESC
        )d, (select @rownum :=0 , @pu:= null ,@rank:=0) a
        )c group by c.user_id;
 """
    return sqlTemplate.format(table_name=table_name, fromTableName = fromTableName)
    

def genAll():
    drop_tmp_table ="""
        drop table if exists tmp_top_5train;
        drop table if exists tmp_top_5test;
        drop table if exists tmp_top_5submit;            
        """
    return (gen_tmp_table(fromDateTrain, toDateTrain, 'top_5train'), 
            gen_tmp_table(fromDateTest, toDateTest, 'top_5test'),
            gen_tmp_table(fromDateSubmit, toDateSubmit, 'top_5submit'),
            gen(fromDateTrain, toDateTrain, 'user_%s_train' %featueName), 
            gen(fromDateTest, toDateTest, 'user_%s_test' %featueName),
            gen(fromDateSubmit, toDateSubmit, 'user_%s_submit' %featueName),
            drop_tmp_table)
    
if __name__ == '__main__':
    for sql in genAll():
        print sql