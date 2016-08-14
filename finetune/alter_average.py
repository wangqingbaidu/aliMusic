# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 19, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import sys

def get_ID_average(ifile_name = None):
    assert ifile_name
    output = {}
    f = open(ifile_name).readlines()
    for item in f:
        user_id, play_times, _ = tuple(item.split(','))
        if output.has_key(user_id):
            output[user_id].append(float(play_times))
        else:
            output[user_id] = [float(play_times)]
            
    for key in output.keys():
        output[key] = sum(output[key]) / len(output[key])
    
    return output

def get_output_file(ifile_name = None, ratio = 1):
    assert ifile_name
    ofile_name= ifile_name.replace('.csv', '_avg.csv')
    avg = get_ID_average(ifile_name)
    fi = open(ifile_name).readlines()
    fo = open(ofile_name, 'w')
    for item in fi:
        user_id, play_times, pdate = tuple(item.split(','))
        play_times = avg[user_id] * ratio
        fo.write('%s,%s,%s' %(user_id, int(play_times), pdate))
        
    fo.close()
    
if __name__ == "__main__":
    assert len(sys.argv) == 2
    ifile_name = sys.argv[1]
    get_output_file(ifile_name=ifile_name)
    