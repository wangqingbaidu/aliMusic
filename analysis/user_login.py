# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jun 5, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import pymysql
import datetime, time
import random
import matplotlib                                                                        
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def user_login():
    mysql_cn= pymysql.connect(host='10.25.0.119', port=3306,user='root', passwd='111111', db='music')
    artist_list = pd.read_sql('select * from artist_list', mysql_cn).values.tolist()
    artist_list = [(x[0], int(x[1])) for x in artist_list]
    cur = mysql_cn.cursor()
    getTotal = '''
        SELECT COUNT(*) from user_actions join songs
        on user_actions.song_id = songs.song_id
        where ds >= '{fromDate}' and ds <= '{toDate}' 
        AND songs.artist_id = '{artist_id}' 
        and action_type = '1';
    '''
    dropTemp = '''    
        DROP table IF EXISTS login_tmp;
    '''
    createTemp = '''
        CREATE TABLE login_tmp as
            SELECT user_id FROM(
            SELECT user_id, COUNT(*) as login FROM(
            SELECT user_id, ds FROM user_actions join songs
            on user_actions.song_id = songs.song_id
            where ds >= '{fromDate}' and ds <= '{toDate}' 
            AND songs.artist_id = '{artist_id}' 
            GROUP BY user_id, ds)a
            GROUP BY user_id)b
            WHERE login >= {threshold};
    '''
    createTempIndex = '''
        CREATE INDEX IDX_login_tmp on login_tmp(user_id);
    '''
    getUserLogin = '''
        SELECT count(*) FROM user_actions join songs
        on user_actions.song_id = songs.song_id
        where ds >= '{fromDate}' and ds <= '{toDate}' 
        AND songs.artist_id = '{artist_id}' 
        and action_type = '1' and user_id IN  
            (SELECT * FROM login_tmp);
    '''
    recentNdays = 10
    deltaDay = 30
    varianceDay = 3
    DataStartDelta = datetime.timedelta(days = -recentNdays - deltaDay + 2)
    fromDateStart = datetime.datetime.strptime('20150830', "%Y%m%d").date() + DataStartDelta
    fromDateStart = fromDateStart.strftime('%Y%m%d')
#     print fromDateStart
    toDateDelta = datetime.timedelta(days= deltaDay - 1)
#     artist_id = '4b8eb68442432c242e9242be040bacf9'
    artist_count = 0
    assert varianceDay <= deltaDay
    dateNow = time.strftime('%Y%m%d')
    for artist_id, plays in artist_list: 
        res = []
        if artist_id != 'cd5ce8f47e50971ddb629d86a0bc34f2':
            continue
        artist_count += 1
        for i in range(recentNdays):
            print 'Getting artist:%s  No.%d' %(artist_id, i + 1) 
            fromDateDelta = datetime.timedelta(days = random.randint(100, 180 - deltaDay))
            fromDateDelta = datetime.timedelta(days = i)
            fromDate = datetime.datetime.strptime(fromDateStart, "%Y%m%d").date() + fromDateDelta
            toDate = fromDate + toDateDelta
            fromDate = fromDate.strftime('%Y%m%d')
            toDate = toDate.strftime('%Y%m%d')
            print fromDate, toDate
            cur.execute(getTotal.format(fromDate = fromDate, toDate = toDate,
                                        artist_id = artist_id))
            for item in cur:
                total = item[0]
            cur.execute(dropTemp)
            cur.execute(createTemp.format(fromDate = fromDate, toDate = toDate, 
                                          artist_id = artist_id, threshold = deltaDay / 2))
            cur.execute(createTempIndex)
            cur.execute(getUserLogin.format(fromDate = fromDate, toDate = toDate,
                                            artist_id = artist_id))
            for item in cur:
                expectTotal = item[0]
            print total, expectTotal
            res.append((float(total), expectTotal))
        ratio_df = pd.DataFrame([expectTotal / total for total, expectTotal in res])
        ratio_df.columns = ['%s:%d' %(artist_id, plays)]
        ratio_df.plot()
        fig = plt.gcf()
        fig.savefig('./img/ratio_No{No:0>2}_{artist_id}_{deltaDay:0>2}.png'.format(No = artist_count, 
                                                                        artist_id = artist_id, 
                                                                        deltaDay =deltaDay))
        ratio_df.to_csv('./data/ratio_No{No:0>2}_{artist_id}_{deltaDay:0>2}.csv'.format(No = artist_count, 
                                                                        artist_id = artist_id, 
                                                                        deltaDay =deltaDay), 
                        header =False, 
                        index=False)
        ratio = [expectTotal / total for total, expectTotal in res]
        variance_df = pd.DataFrame([[abs(np.mean(np.array(ratio[e:varianceDay + e])) - ratio[e]),
                                    np.var(ratio[e:varianceDay + e])]
                                    for e in range(len(ratio) - varianceDay + 1)])
        variance_df.columns = ['%s:Mean-ratio' %(artist_id), 'Variance']
        variance_df.plot()
        fig = plt.gcf()
        fig.savefig('./img/variance_No{No:0>2}_{artist_id}_{deltaDay:0>2}.png'.format(No = artist_count, 
                                                                        artist_id = artist_id, 
                                                                        deltaDay =deltaDay))
        variance_df.to_csv('./data/variance_No{No:0>2}_{artist_id}_{deltaDay:0>2}.csv'.format(No = artist_count, 
                                                                        artist_id = artist_id, 
                                                                        deltaDay =deltaDay), 
                        header =False, 
                        index=False)
#     df = pd.DataFrame(res)
#     df.to_csv('login_fre.csv', header =False, index=False)
    mysql_cn.close()
    
if __name__ == '__main__':
    user_login()
