# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 21, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
from genData.gen_data_all import genAll as genAllData

from utils import gen_u_a_d_keys, gen_u_s_d_keys
from gen_y_features import gen_y_features_all


def gen_artist_label_map():
    sql = """
    DROP table if EXISTS artist_label_map;
    CREATE TABLE artist_label_map as
    SELECT artist_id, @rank:=@rank+1 as label from songs, (SELECT @rank:=0)a
    GROUP BY artist_id;
    """
    
    return [sql]

if __name__ == '__main__':
    ignore_date = True
    keys_type = 'u_a_d'
    sql = ""
    
    for i in gen_artist_label_map():
        sql += i
    
    if keys_type == 'u_a_d':
        for i in gen_u_a_d_keys.genAll():
            sql += i
        
        for i in gen_y_features_all.gen_y_features_all(keys_type, ignore_date):
            sql += i
            
        for i in genAllData(keys_type, ignore_date):
            sql += i
    elif keys_type == 'u_s_d':
        for i in gen_u_s_d_keys.genAll():
            sql += i
            
        for i in gen_y_features_all.gen_y_features_all(keys_type, ignore_date):
            sql += i
            
        for i in genAllData(keys_type, ignore_date):
            sql += i
    
    print sql
        
