# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 15, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''

featueName = 'keys'
from gen_X_features import fromDateTrain, toDateTrain, fromDateTest, toDateTest, fromDateSubmit, toDateSubmit

def gen(fromDate = None, toDate = None, table_name = None):
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    SELECT user_id, song_id, 
    date_format(date_add(str_to_date(ds, '%Y%m%d'), interval 2 month), '%Y%m%d') as ds 
    from user_actions 
    where action_type = '1' and ds>='{fromDate}' and ds <= '{toDate}'
    GROUP BY user_id, song_id, ds
    order by ds desc;
    create index IDX_{table_name} on {table_name}(user_id);
 """
    return sqlTemplate.format(table_name=table_name, fromDate=fromDate, toDate=toDate)
    

def genAll():
    '''
        This function is used to generate keys which is used as the origin table of 
        left join on train, test, submit dataset.
    '''
    return (gen(fromDateTrain, toDateTrain, 'user_%s_train' %featueName), 
            gen(fromDateTest, toDateTest, 'user_%s_test' %featueName),
            gen(fromDateSubmit, toDateSubmit, 'user_%s_submit' %featueName))
    
if __name__ == '__main__':
    for sql in genAll():
        print sql