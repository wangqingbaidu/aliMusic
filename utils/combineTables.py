# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 15, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''

def gen(ckeys = None, tables = None, columnCondition = None, table_name = None):
    assert type(columnCondition) == str or \
           type(columnCondition) == list or \
           type(columnCondition) == tuple
           
    scolumns = ""
    left_join = ""
    if type(columnCondition) == str:
        scolumns = ',\n\t'.join([key + '.' + col + ' as ' + key +'_'+ col for key in tables.keys() 
                            for col in tables[key] if col != columnCondition])
        
        left_join = '\nleft join '.join(key + ' on ({0}.' + columnCondition + '=' + 
                                      key + '.' + columnCondition + ')' for key in tables.keys()).format(ckeys)
    else:
        scolumns = ',\n\t'.join([key + '.' + col + ' as ' + key +'_'+ col for key in tables.keys() 
                            for col in tables[key] if not col in columnCondition])
        
        left_join = '\n\tleft join '.join(key + ' on (' + ' and '.join([
            '{0}.' + condition + '=' + key + '.' + condition for condition in columnCondition
            ]) + ')' for key in tables.keys()).format(ckeys)
        
    sqlTemplate = """
    drop table if exists {table_name};
    create table {table_name} as
    select 
    \t{scolumns} 
    from
    \t{ckeys} left join {left_join};
 """
    return sqlTemplate.format(table_name = table_name, scolumns=scolumns, ckeys=ckeys, left_join=left_join)
    

if __name__ == '__main__':
    table_dic = {
        'user_day_avg_play_test':['user_id', 'day_avg_play'], 
        'user_day_avg_download_test':['user_id', 'day_avg_download'], 
        'user_day_avg_collect_test':['user_id', 'day_avg_collect'],
        'user_top_5period_collect_test':['user_id','rank1', 'rank2', 'rank3', 'rank4', 'rank5'], 
        'user_top_5period_download_test':['user_id','rank1', 'rank2', 'rank3', 'rank4', 'rank5'], 
        'user_top_5period_play_test':['user_id','rank1', 'rank2', 'rank3', 'rank4', 'rank5']}
    
    print gen('user_keys_test', table_dic, columnCondition=['user_id'],table_name='user_X_test')
    
    table_dic = {
        'y_test_user':['user_id', 'song_id', 'play_times']
    }
     
    print gen('user_keys_test', table_dic, columnCondition=('user_id', 'song_id', 'ds'),table_name='user_y_test')


