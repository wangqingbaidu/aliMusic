# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 15, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import datetime

featureName = 'day_avg_'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit

def gen(fromDate = None, toDate = None, table_name = None, action_type = None):
    assert action_type in [1, 2, 3]
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    select user_id, count(*) / {total_days} as day_avg from user_actions 
    where action_type = '{action_type}' and ds >= '{fromDate}' and ds <= '{toDate}'
    group by user_id;
    create index IDX_{table_name} on {table_name}(user_id);
 """
    total_days = (datetime.datetime.strptime(toDate, "%Y%m%d").date() - 
    datetime.datetime.strptime(fromDate, "%Y%m%d").date()).total_seconds() / (24 * 60 * 60) + 1
    return sqlTemplate.format(table_name = table_name, total_days = int(total_days), 
                              fromDate = fromDate, toDate = toDate, action_type = action_type)

def genAll():
    res = []
    action_type_dic = {1:'play', 2:'download', 3:'collect'} 
    for i in range(3):
        fea_name = featureName + action_type_dic[i + 1] 
        res += [gen(fromDateTrain, toDateTrain, 'user_%s_train' %fea_name, i + 1), 
            gen(fromDateTest, toDateTest, 'user_%s_test' %fea_name, i + 1),
            gen(fromDateSubmit, toDateSubmit, 'user_%s_submit' %fea_name, i + 1)]
    return res
    
if __name__ == '__main__':
    for sql in genAll():
        print sql