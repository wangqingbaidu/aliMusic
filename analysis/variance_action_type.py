# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jun 4, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt

def variance_download():
    mysql_cn= pymysql.connect(host='10.25.0.119', port=3306,user='root', passwd='111111', db='music')
    df = pd.read_sql('''
    select SUM(before_plays) as sbefore_plays, SUM(after_plays) as safter_plays from user_download_compare
    GROUP BY user_id
    limit 500;
    ''', mysql_cn)
    df.plot(kind='bar')
    plt.show()
    
    mysql_cn.close()
    
    
if __name__ == '__main__':
    variance_download()
