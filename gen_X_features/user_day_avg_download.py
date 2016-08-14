# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 15, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import datetime

featueName = 'day_avg_download'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit

def gen(fromDate = None, toDate = None, table_name = None):
    sql = ''
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    select user_id, count(*) / {total_days} as {featueName} from user_actions 
    where action_type = '2' and ds >= '{fromDate}' and ds <= '{toDate}'
    group by user_id;
 """
    total_days = (datetime.datetime.strptime(toDate, "%Y%m%d").date() - 
    datetime.datetime.strptime(fromDate, "%Y%m%d").date()).total_seconds() / (24 * 60 * 60) + 1
    return sqlTemplate.format(table_name = table_name, total_days = int(total_days), 
                              fromDate = fromDate, toDate = toDate, featueName = featueName)
    
    return sql

def genAll():
    return (gen(fromDateTrain, toDateTrain, 'user_%s_train' %featueName), 
            gen(fromDateTest, toDateTest, 'user_%s_test' %featueName),
            gen(fromDateSubmit, toDateSubmit, 'user_%s_submit' %featueName))
    
if __name__ == '__main__':
    for sql in genAll():
        print sql