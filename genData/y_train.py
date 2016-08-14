# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 15, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''

from utils.combineTables import gen as gen_y_train

def gen(ckeys = None, conditions = None, table_name = None):
    assert ckeys and conditions, table_name
    
    table_dic = {
        'y_train_user':['play_times']
    }
    
    return gen_y_train(ckeys=ckeys, tables = table_dic, columnCondition=conditions,table_name=table_name)
    
if __name__ == '__main__':
    print gen('user_keys_train', ['user_id', 'song_id'],'user_y_train')
