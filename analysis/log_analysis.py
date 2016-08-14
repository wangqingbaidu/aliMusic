# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jul 12, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
logf = open('log').readlines()
dic = {}
while logf:
    params, result = (logf[0], logf[1])
    r2 = params.split('-')[13]
    p = params.split('-')[17]
    key = '%.2f%d' %(float(r2), int(p))
    score0 = float(result.split(' ')[2])
    score1 = float(result.split(' ')[5])
    if dic.has_key(key):
        dic[key].append(float(score1) / float(score0))
    else:
        dic[key] = [float(score1) / float(score0)]
    logf = logf[2:]
    
dl = sorted(dic.items(), 
            cmp = lambda x,y: -cmp(sum(x) / len(x),sum(y) / len(y)), 
            key = lambda k: k[1])

for x in dl:
    print x[0], sum(x[1]) / len(x[1])