# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jul 13, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import datetime
import pymysql
import pickle
from datetime import date

class ArtistPlaysCycle:
    def __init__(self,
                 start_date = '20150301',
                 end_date = '20150630',
                 ifDebug = False,
                 ifSaveAnalysis = True,
                 host='10.25.0.118', port=3306,user='root', passwd='111111', db='alimusic'):
        self.end_date = end_date
        self.start_date = start_date
        self.ifDebug = ifDebug
        self.ifSaveAnalysis = ifSaveAnalysis
        self.mysql_cn= pymysql.connect(host=host, port=port,user=user, passwd=passwd, db=db)
        
    def __get_week_index(self, data):
        week_index = {}
        data = sorted(data, key = lambda x:x[0], reverse=True)
        for i in range(len(data)):
            week_index[int(data[i][1])] = i + 1
            
        return week_index[6] + week_index[0]    
    
    def __gen_data_set(self):
        self.artist_map = {}
        artist_list = [(x[0], x[1]) 
                      for x in pd.read_sql('select artist_id, plays from artist_list order by artist_id desc', 
                                           self.mysql_cn).values.tolist()]
        
        for artist, _ in artist_list:
            sqlTemplate = '''
            select plays, wd from clean_artist_play_with_week
            where artist_id = '{artist_id}' and ds >= '{start_date}' and ds <= '{end_date}'
            order by ds;
            '''.format(artist_id = artist, start_date = self.start_date, end_date = self.end_date)
            
            data = pd.read_sql(sqlTemplate, self.mysql_cn).values.tolist()
            while len(data) >= 7:
                test = data[:7]
                max_index = max(test, key = lambda x : x[0])[1]
                min_index = min(test, key = lambda x : x[0])[1]
#                 test.remove(max(test, key = lambda x : x[0]))
#                 test.remove(min(test, key = lambda x : x[0]))
                max_type = -1
                min_type = -1
                week_index = self.__get_week_index(test)
                if  week_index <= 4:
                    max_type = 100
                else:
                    max_type = 101
                 
                if week_index >= 12:
                    min_type = 100
                else:
                    min_type = 101
#                     
#                 if  max_index == 0 or max_index == 6:
#                     max_type = 100
#                 else:
#                     max_type = 101
#                 
#                 if min_index == 0 or min_index == 6:
#                     min_type = 100
#                 else:
#                     min_type = 101
                    
                if self.artist_map.has_key(artist):
                    self.artist_map[artist]['max_day'].append(max_index)
                    self.artist_map[artist]['min_day'].append(min_index)
                    self.artist_map[artist]['max_type'].append(max_type)
                    self.artist_map[artist]['min_type'].append(min_type)
                else:
                    self.artist_map[artist] = {}
                    self.artist_map[artist]['max_day'] = [max_index]
                    self.artist_map[artist]['min_day'] = [min_index]
                    self.artist_map[artist]['max_type'] = [max_type]
                    self.artist_map[artist]['min_type'] = [min_type]
                data = data[7:]
            if self.ifDebug:                
                print artist, 'got!' 
                print self.artist_map[artist]
    
    def getDataSet(self):
        try:
            return self.artist_map
        except:
            self.__gen_data_set()
            return self.artist_map
    
    def setDataSet(self, ds):
        self.artist_map = ds
           
    def __gen_artist_std(self):
        data_set = self.getDataSet()
        save_analysis = []
        for artist in data_set:
            max_day_std = np.std(data_set[artist]['max_day'])
            min_day_std = np.std(data_set[artist]['min_day'])
            max_type_std = np.std(data_set[artist]['max_type'])
            min_type_std = np.std(data_set[artist]['min_type'])
            self.artist_map[artist]['std'] = [max_day_std, min_day_std, max_type_std, min_type_std]
            if self.ifDebug:
                print '%s,%f,%f,%f,%f' %(artist, max_day_std, min_day_std, max_type_std, min_type_std)
            if self.ifSaveAnalysis:
                save_analysis.append([artist, float(max_day_std), float(min_day_std), 
                                              float(max_type_std), float(min_type_std)])
        if self.ifSaveAnalysis:
            save_analysis = pd.DataFrame(save_analysis,columns=['artist_id', 'max_day', 
                                                                'min_day', 'max_type', 'min_type'])
            save_analysis.to_csv('cycle_analysis.csv', index = False)
            
    def getFilteredArtist(self):
        filtered_artist = []
        data_set = None
        try:
            pkl_file = open('artist_plays_cycle%s_%s.pkl' %(self.start_date, self.end_date), 'rb')
            data_set = pickle.load(pkl_file)
            self.setDataSet(data_set)
            pkl_file.close()
        except:
            data_set = self.getDataSet()
            output = open('artist_plays_cycle%s_%s.pkl'%(self.start_date, self.end_date), 'wb')
            pickle.dump(data_set, output)
            output.close()

        self.__gen_artist_std()
        assert data_set
        for artist in data_set:
            if self.artist_map[artist]['std'][1] == 0 or \
                self.artist_map[artist]['std'][3] == 0:
                filtered_artist.append([artist, data_set[artist]])       
#         for artist in data_set:
#             if self.artist_map[artist]['std'][0] == 0 or \
#                 self.artist_map[artist]['std'][1] == 0 or \
#                 self.artist_map[artist]['std'][2] == 0 or \
#                 self.artist_map[artist]['std'][3] == 0:
#                 filtered_artist.append([artist, data_set[artist]])
        return filtered_artist
        
    def getResultFile(self):
        results = []
        for artist, data in self.getFilteredArtist():
            if data['std'][0] == 0:
                if (artist, int(data['max_day'][0])) not in results:
                    results.append((artist, int(data['max_day'][0])))
            if data['std'][1] == 0:
                if (artist, int(data['min_day'][0])) not in results:
                    results.append((artist, int(data['min_day'][0])))
                
    #    if weekends
            if (data['std'][0] != 0 and data['std'][2] == 0 and data['max_type'][0] == 100) or \
                (data['std'][0] != 0 and data['std'][3] == 0 and data['min_type'][0] == 100):
                if (artist, 0) not in results:
                    results.append((artist, 0))
                if (artist, 6) not in results:
                    results.append((artist, 6))
        pd.DataFrame(results).to_csv('cycle%s_%s.txt' %(self.start_date, self.end_date), 
                                     index = False, header=False)
        
if __name__ == '__main__':
    apc = ArtistPlaysCycle(start_date='20150530',
                           ifDebug=False,
                           ifSaveAnalysis = True)
