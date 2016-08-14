# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 19, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import os, sys
def do_multi_by_ratio(fp = None, ratio = 1):
    assert fp
    if not os.path.isfile(fp):
        print 'File %s not exists!' %fp
        return
    df = pd.read_csv(fp, names=['artist_id', 'plays', 'ds'])
    df['plays'] = df['plays'].astype(int) / ratio
    df['plays'] = df['plays'].astype(int)
    save_file = '{0}_multi_by_{1:.2f}.csv'.format(fp.replace('.csv', ''), float(ratio))
    print 'Alter file save to %s' %save_file
    df.to_csv(save_file, header = False, index = False)

def multi_by_ratio(ifile = None, ratio = 1):
    assert ifile
    if type(ifile) == str:
        do_multi_by_ratio(ifile, ratio)
    else:
        for fp in ifile:
            do_multi_by_ratio(fp, ratio)
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print '''Please input file(s) and ratio to be alter!\nUsage ratio, list of filename'''
        exit()
    ratio = 0
    try:
        ratio = float(sys.argv[1])
    except:
        print 'Ratio must be float or int!'
        exit()
    ifile = sys.argv[2:]
    multi_by_ratio(ifile, ratio)
    
