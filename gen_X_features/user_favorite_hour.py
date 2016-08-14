# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 26, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
featueName = 'favorite_hour'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit

def gen(fromDate = None, toDate = None, table_name = None):
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    select c.user_id,
    {hour_matrix}
    from(
    SELECT user_id, FROM_UNIXTIME( cast(gmt_create AS UNSIGNED), '%H' ) AS hour_of_day, count(*) AS times 
    FROM user_actions 
    where ds >= '{fromDate}' and ds <= '{toDate}'
    GROUP BY user_id, hour_of_day
    )c group by c.user_id;
    create index IDX_{table_name} on {table_name}(user_id);
 """
    hour_matrix = ",\n\t".join(
        ["max(if(c.hour_of_day='{index}',times,NULL) )as fav_hour{index}".format(index = '{:0>2}'.format(index)) 
        for index in range(24)])
    return sqlTemplate.format(table_name = table_name, fromDate = fromDate, toDate = toDate,
                              hour_matrix = hour_matrix)
    

def genAll():
    return (gen(fromDateTrain, toDateTrain, 'user_%s_train' %featueName), 
            gen(fromDateTest, toDateTest, 'user_%s_test' %featueName),
            gen(fromDateSubmit, toDateSubmit, 'user_%s_submit' %featueName))
    
if __name__ == '__main__':
    for sql in genAll():
        print sql