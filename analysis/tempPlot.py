# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jun 13, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

def analysis():
    mysql_cn= pymysql.connect(host='10.25.0.119', port=3306,user='root', passwd='111111', db='music')
    df = pd.read_sql('''
    SELECT COUNT(*) as plays, ds from user_actions JOIN songs
    on user_actions.song_id = songs.song_id
    WHERE ds >= '20150805' AND ds <= '20150830' AND action_type = '1' 
    AND artist_id = 'c026b84e8f23a7741d9b670e3d8973f0'
    GROUP BY artist_id, ds 
    ORDER BY ds
    '''.format(),mysql_cn)
    X = np.array([i for i in range(26)])
    df.columns = ['plays', 'ds']
    y = df['plays'].values
    print X, y
    model = LinearRegression()
    model.fit(X.reshape(X.shape[0], 1), y.reshape(y.shape[0]))
    x = np.array([i for i in range(26, 50)])
    Y = model.predict(x.reshape(x.shape[0], 1))
    df = pd.DataFrame(Y)
    print Y
    df.plot()
    plt.show()
    
    mysql_cn.close()
    
    
if __name__ == '__main__':
    analysis()
