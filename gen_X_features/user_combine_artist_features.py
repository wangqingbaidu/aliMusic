# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 30, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import datetime

featueName = 'artist_features'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit

def gen(fromDate = None, toDate = None, table_name = None, table_type = None):
    assert table_type and table_name
    names = open('artist_features_names').readlines()[0].split(',')
    columns = ',\n\t'.join('artist_{table_type}_one_month.'.format(table_type = table_type) + name
                             for name in names)
    sqlTemplate = """
    drop table if exists {table_name}{table_type};
    create table {table_name}{table_type} as
    SELECT
    {columns}
    FROM 
    user_keys_{table_type} LEFT JOIN artist_{table_type}_one_month
    on user_keys_{table_type}.artist_id = artist_{table_type}_one_month.artist_id AND
    user_keys_{table_type}.ds = artist_{table_type}_one_month.gmt_date;
 """
    return sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate,
                              table_type = table_type, columns = columns)
    

def genAll():
    table_types = ['train', 'test', 'submit']
    res = []
    for table_type in table_types:
        res.append(gen(fromDateTrain, toDateTrain, 'user_%s_' %featueName, table_type))
    return res
    
if __name__ == '__main__':
    for sql in genAll():
        print sql