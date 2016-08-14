#!/usr/bin/env python
############################
###Author:Vincent.Y
###Date  : 2016.06.17
############################

import pandas as pd
import numpy as np
import datetime

def evaluate(y_true,y_pred):
    sig=np.sqrt(np.sum(y_true))
    data=[]
    for i,j in zip(y_true,y_pred):
        if i==0:
            continue
        data.append(np.power((int(j)*1.0-i)/i,2))
    delta=np.sqrt(np.mean(np.array(data)))
    #print sig,delta
    return (1-delta)*sig
def get_min_error_res(play_res):
    res_sum = 0
    res_sum_2 = 0
    for res in play_res:
        if res < 1: continue
        res_sum += 1.0/res
        res_sum_2 += 1.0/(res*res)
    if res_sum == 0: return 0
    return res_sum / res_sum_2
def getdf(df,artist_id,start,end):
    return (df.ix[(df.artist_id==artist_id) & (df.gmt_date>int(start)) & (df.gmt_date<int(end)),:]).sort_values(by="gmt_date",ascending=1)

if __name__ == '__main__':
    data=pd.read_csv("artist_play.csv")
    all_start=20150401
    all_end=20150701
    artists=data.artist_id.unique()
    for artist_id in artists:
        x=datetime.datetime.strptime(str(all_start),"%Y%m%d")
        y=x+datetime.timedelta(days=61)
        while int(x.strftime("%Y%m%d"))<=all_end:
            split=x
            xl=0
            for i in range(2,60):
                start=int((split-datetime.timedelta(days=i)).strftime("%Y%m%d"))
                xdf=getdf(data,artist_id,start,int(x.strftime("%Y%m%d")))
                if len(xdf)==xl:
                    break
                xl=len(xdf)
                print start, xdf.artist_moving
                value=get_min_error_res(xdf.artist_moving)
                df=getdf(data,artist_id,int(x.strftime("%Y%m%d")),int(y.strftime("%Y%m%d")))
                print ("%s,%s,%s,%s,%s,%s,%s")%(artist_id,i,x.strftime("%Y%m%d") ,evaluate(df.artist_moving,[value]*len(df.artist_moving)),value,get_min_error_res(df.artist_moving),evaluate(df.artist_moving,[get_min_error_res(df.artist_moving)]*len(df.artist_moving)))
            x=x+datetime.timedelta(days=1)
            y=x+datetime.timedelta(days=61)
#             exit()