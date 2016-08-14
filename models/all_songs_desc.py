# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jul 3, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import numpy as np
import pymysql
import matplotlib                                                                        
# from sklearn.preprocessing.data import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import PolynomialFeatures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import random
random.seed(219)

data2use = {True:'clean_user_actions_with_artists', False:'user_actions_with_artists'}
maxt = {True:None, False:'smaller than maxt'}
lent = {True:None, False:'smaller than lent'}
                
class NewSongDecay:
    def __init__(self, 
                 toDate = None, 
                 bySelfModel = False,
                 new_song_day_use = 61, 
                 me_day_use = 14, 
                 use_clean = True,
                 first_use = False,
                 ifTestOnline = False,
                 ifDebug = True,
                 ifPlot = True,
                 ifPlotTrain = False,
                 ifShowEachArtistScore = False,
                 host='10.25.0.118', port=3306,user='root', passwd='111111', db='music'):
        assert toDate
        self.ifDebug = ifDebug
        self.ifTestOnline = ifTestOnline
        self.ifShowEachArtistScore = ifShowEachArtistScore
        self.ifPlot = ifPlot
        self.ifPlotTrain = ifPlotTrain,
        self.toDate = toDate
        self.bySelfModel = bySelfModel
        self.new_song_from_date = (datetime.datetime.strptime(self.toDate, '%Y%m%d') + 
                                   datetime.timedelta(days = -new_song_day_use + 1)).strftime('%Y%m%d')
        self.me_day_use = me_day_use
        self.me_from_date = (datetime.datetime.strptime(self.toDate, '%Y%m%d') + 
                        datetime.timedelta(days = -me_day_use + 1)).strftime('%Y%m%d')
                        
        
        tfgapday = datetime.timedelta(days=2)
        ttgapday = datetime.timedelta(days= 61)
        self.tfromDate = (datetime.datetime.strptime(self.toDate, "%Y%m%d").date() + 
                          tfgapday).strftime('%Y%m%d')
        self.ttoDate = (datetime.datetime.strptime(self.toDate, "%Y%m%d").date() + 
                        ttgapday).strftime('%Y%m%d')
                        
        self.use_clean = use_clean
        self.first_use = first_use
        
        self.decay_artist = []
        self.X_additional_features = None
        self.mysql_cn= pymysql.connect(host=host, port=port,user=user, passwd=passwd, db=db)
    
    def __del__(self):
        if self.mysql_cn:
            self.mysql_cn.close()
            
#     def __transform_X(self, x):
#         return 1.0/(1+np.log2(x)/np.log2(100))
#     
#     def __transform_y(self, y, max_y):
#         return y / max_y
#     
#     def __inverse_transform_y(self, y, max_y):
#         return y * max_y

    def __transform_X(self, x):
        #zzw
        #return 1.0/(1+np.log2(x)/np.log2(100)) #52989.9067967
        #return 1.0/x #52937.9429992
        
        #return 1.0 / (np.power(x,0.5)) #52974.038936
        #return 1.0 / (1+np.power(0.5, x)) #52906.5056225
        #return 1.0 / (1+np.power(2, x)) #52906.5056225
        #return 1.0 / (np.power(0.5, x)) #52876.5004025
        #return 1.0 / (1+np.power(0.5, x)) #52906.5056225
        #return 1.0 / (np.power(x,0.1)) #52998.7063974
        #return 1.0 / (np.power(x,0.01)) #53003.4783104
        #return 1.0 / (np.power(x,0.1)+1) #53003.0965201
        #return 1.0 / (np.power(x,0.01)+1) #53003.8869269
        #return 1.0 / (np.power(0.01, x)+1) #52889.6206326
        #return 1.0 / (np.power(0.01, x))  #52889.6206326
        #return 0 * x #52877.0154679
        #Polynomial fit
        return np.power(x, 0.1)
        #return np.log2(x)
        #return 1.0/(1+np.log2(x)/np.log2(100)) #52989.9067967
        #return 1/x
        #aaaaaaaa
        #power function
        #return x
        
        
    def __transform_y(self, y, max_y):
        #zzw
        #return y / max_y
        #power function
        #ln(y+1) = kx+b
        return y/max_y
    
    def __inverse_transform_y(self, y, max_y):
        #zzw
        return y * max_y
        #power function
        #return np.power(100, y) * max_y- 1
            
    def getOnlineSQL(self):
        return self.genDependence() + self.genPeriodME() + self.genNewSongOutBaseline()
    
    def getResultSQL(self):
        return self.genResultTable()
    
    def genDependence(self):
        sqlTemplate = '''
        drop table if exists zzw_all_keys;
        create table zzw_all_keys as
        select c.artist_id,to_char(dateadd(to_date(c.ds,"yyyymmdd"),61,"dd"),"yyyymmdd") as ds from (
            select artist_id,ds from user_song
            where ds>"20150301" and ds<"20150831" group  by artist_id,ds
        )c;
        
        drop table if exists zzw_new_songs;
        create table zzw_new_songs as 
          select a.*, b.publish_time from 
              {data2use} a
          join (
              select song_id, publish_time from songs 
              where publish_time >= '20150301' and publish_time <= '20150830'
          )b
          on a.song_id = b.song_id;
          
        drop table if exists new_songs_plays;
        create table new_songs_plays as
          select concat(artist_id, publish_time) as artist_id, ds, count(1) as plays from zzw_new_songs
          where action_type = '1'
          group by artist_id, publish_time, ds;
        '''
        if self.first_use:
            return sqlTemplate.format(data2use = data2use[self.use_clean])
        else:
            return ''
     
    def genPeriodME(self):
        
        if self.ifDebug:
            print self.new_song_from_date, self.me_from_date, self.toDate
        sqlTemplate = '''
        drop table if exists period_new_songs_out_me;
        create table period_new_songs_out_me as
          select artist_id,sum(1/plays)/sum(1/plays/plays) as plays
          from(
              select song_out.artist_id,song_out.ds,count(1) as plays
              from (
              select * from (
                  select cua.*, a.ot from
                  {data2use} cua
                  left outer join(
                    select song_id, '1' as ot from songs
                    where publish_time >= '{new_song_from_date}' and publish_time <= '{toDate}'
                    group by song_id
                  )a on cua.song_id = a.song_id
                )tmp where tmp.ot is null
              )song_out
              where song_out.action_type=1                
                 and song_out.ds >= "{me_from_date}"
                 and song_out.ds <= "{toDate}"    
              group by song_out.artist_id,song_out.ds
          )b group by artist_id;
        '''
        return sqlTemplate.format(new_song_from_date = self.new_song_from_date, 
                                  toDate = self.toDate,
                                  me_from_date = self.me_from_date,
                                  data2use = data2use[self.use_clean])
     
    def genNewSongOutBaseline(self):
        sqlTemplate = '''
        drop table if exists new_song_out_baseline;
        create table new_song_out_baseline as
        select k.*, me.plays from(
            select * from zzw_all_keys
            where ds >= '{new_song_from_date}' and ds<= '{toDate}'
        )k 
        left outer join
            period_new_songs_out_me me
        on k.artist_id = me.artist_id;
        '''
        return sqlTemplate.format(new_song_from_date = self.tfromDate, 
                                  toDate = self.ttoDate)
    
    def genResultTable(self):
        sqlResult = '''
        DROP TABLE IF EXISTS zzw_new_songs_tendency;
        CREATE table zzw_new_songs_tendency as
            SELECT b.artist_id, b.ds,  if((b.plays + d.plays) is NULL, b.plays,  (b.plays + d.plays))as plays FROM
            new_song_out_baseline b
            LEFT OUTER JOIN new_songs_incr d
        on (b.artist_id = d.artist_id and b.ds = d.ds);
        '''
        sqlTemplate = '''        
        select sum(f.fscore) as fscore_best_1 from(
          select  artist_id,sqrt(sum(e.target)) * (1-sqrt(avg(pow((e.plays-e.target)/e.target,2)))) as fscore
          from(
            select c.artist_id,c.ds,c.target,d.plays from(
                select t.artist_id,t.ds,t.target
                from (
                    select artist_id, ds, count(1) as target from
                    user_actions_with_artists
                    where action_type = 1
                    and ds >= '{new_song_from_date}'
                    and ds <= '{toDate}'
                    group by artist_id, ds
                )t
            )c
            left outer join
                zzw_new_songs_tendency d on(c.artist_id=d.artist_id and c.ds = d.ds)
          )e group  by artist_id
        )f;
        '''
        if self.ifTestOnline:
            return sqlResult + sqlTemplate.format(new_song_from_date = self.tfromDate, 
                                                  toDate = self.ttoDate)
        else:
            return sqlResult
    def setOtherXFeatures(self, X_additional_features = None):
        assert type(X_additional_features) is dict
        self.X_additional_features = X_additional_features
        
    def __gen_data_set(self, max_value_threshold = 1000, train_length_threshold = 30):
        artist_id_list = [(x[0], x[1]) 
                      for x in pd.read_sql('select artist_id, plays from artist_list order by artist_id desc', 
                                           self.mysql_cn).values.tolist()]
        X_train = None
        y_train  = None
        first = False 
        test = []
        
        for artist_id, _ in artist_id_list:
#           Use start parameter because of some songs are listened ahead of artist publish time   
            df = pd.read_sql('''
            SELECT a.artist_id, a.ds, (a.plays - b.plays) as plays from 
            artist_play a
            join
            baseline b
            on a.artist_id = b.artist_id and a.ds = b.ds
            WHERE artist_id = '{artist_id}' and ds >= '{start}'
            order by ds;
            '''.format(artist_id=artist_id, start = artist_id[-8:]),self.mysql_cn)
            
            y_array = df.astype(float).values
            y_index = y_array.argmax()
            max_y = y_array[y_index]
            y = self.__transform_y(y_array[y_index:], max_y).reshape((-1))
            X = self.__transform_X(np.arange(1, y.shape[0] + 1).reshape((-1, 1)))
            
#           If additional X features is set, then It must contains all artist_id features 
            assert (not self.X_additional_features or 
                    (self.X_additional_features and self.X_additional_features.has_key(artist_id)))
            if self.X_additional_features and self.X_additional_features.has_key(artist_id):
                X = np.hstack((X, np.array([self.X_additional_features[artist_id] for _ in range(X.shape[0])])))
                
#           When max plays is low, the artist_id plays jitters. 
            if artist_id[-8:] >= self.new_song_from_date and artist_id[-8:] <= self.toDate:
                test.append((X, y, artist_id, max_y, y_index))
            elif X.shape[0] > train_length_threshold and np.max(df.values) > max_value_threshold:
                if not first:
                    y_train = y
                    X_train = X
                    first = True
                else:
                    y_train = np.hstack((y_train, y))
                    X_train = np.vstack((X_train, X))
            elif self.ifDebug:
                info = [maxt[X.shape[0] > train_length_threshold], lent[np.max(df.values) > max_value_threshold]]
                print "artist_id %s dropped beacause of %s" %(artist_id, ' and '.join([s for s in info if s]))
                
        self.data_set = (X_train, y_train, test)
    
    def getDataSet(self, max_value_threshold = 1000, train_length_threshold = 30):
        try:
            return self.data_set
        except:
            self.__gen_data_set(max_value_threshold = max_value_threshold, 
                                train_length_threshold = train_length_threshold)
            return self.data_set
            
#     def __gen_model(self, model = LinearRegression()):
#         X_train, y_train, _ = self.getDataSet(10000, 60)
#         model.fit(X_train, y_train)
#         if self.ifPlotTrain:
#             y_pred = model.predict(X_train)
#             df = pd.DataFrame(np.hstack((y_train.reshape(-1,1), y_pred.reshape(-1,1))))
#             df.columns = ['Train', 'Predict']
#             df[:60].plot()
#             plt.title('train_all')
#             fig = plt.gcf()
#             fig.savefig('./img/train_all.png')
#             plt.close(fig)
#         self.model = model   
             
    def __gen_model(self, model = LinearRegression()):
        model = Pipeline([('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression(fit_intercept=False))])
        X_train, y_train, _ = self.getDataSet()
        model.fit(X_train, y_train)
#         print "mode coef: ",
#         print model.named_steps['linear'].coef_
        self.model = model
    def getModel(self):
        try:
            return self.model
        except:
            self.__gen_model()
            return self.model
            
    def getTendency(self, end_pred_date = '20151030', result_file_name = 'new_songs_incr.csv'):
        X_all = np.arange(1, 365)
        X_all = X_all.reshape((-1, 1))
        test = self.getDataSet()[-1]
        result = []
        for xt, yt, artist_id, max_y, index in test:
            gapday = datetime.timedelta(days=index) 
            dateFrom = datetime.datetime.strptime(artist_id[-8:], '%Y%m%d')
            X = self.__transform_X(X_all)
            self_X = xt
            if self.X_additional_features and self.X_additional_features.has_key(artist_id):
                X = np.hstack((X, np.array([self.X_additional_features[artist_id] for _ in range(X.shape[0])])))
                self_X = np.hstack((xt, np.array([self.X_additional_features[artist_id] for _ in range(xt.shape[0])])))
            model = None
            if self.bySelfModel and self_X.shape[0] >= 3:
                model = LinearRegression()
                train_num = min(self_X.shape[0], 10)
#                 model.fit(self_X[-train_num:], yt[-train_num:])
                model.fit(self_X, yt)
            else:
                model = self.getModel()
            pred = model.predict(X).reshape((-1,1))
            pred = self.__inverse_transform_y(pred, max_y).reshape((-1))
            
            if self.ifDebug:
                print artist_id, index
            if not artist_id[:-8] in self.decay_artist:
                self.decay_artist.append(artist_id[:-8])    
            for plays in pred.tolist():
                dateNow = (dateFrom + gapday).strftime('%Y%m%d')
                gapday += datetime.timedelta(days=1) 
                if dateNow > end_pred_date:      
                    if self.ifPlot:
                        df = self.__inverse_transform_y(pd.DataFrame(yt), max_y)
                        trueV = np.vstack((df.values, np.zeros((gapday.days, 1))))
                        df = pd.DataFrame(trueV)
                        df['predict'] = pd.DataFrame(pred)
                        df.columns = ['origin', 'predict']
                        if df.shape[0] == 1:
                            continue
                        
                        r2 = r2_score(yt[max(-yt.shape[0], -5):], 
                                      self.__transform_y(pred[max(-yt.shape[0], -5) + yt.shape[0]:yt.shape[0]], 
                                                                 max_y))
                        mse = mean_squared_error(yt[max(-yt.shape[0], -5):], 
                            self.__transform_y(pred[max(-yt.shape[0], -5) + yt.shape[0]:yt.shape[0]],
                                               max_y)) * 10000
                        df.plot()
                        plt.title('%s-%.2f r2:%.2f mse:%.2f' %(artist_id[:8], plays / max_y * 100, r2, mse))
                        fig = plt.gcf()
                        fig.savefig('./img/{toDate}_{artist_id}.png'.format(artist_id = artist_id, toDate = self.toDate))
                        plt.close(fig)
                    break
                result.append([artist_id[:-8], dateNow, float(plays)])
        result = pd.DataFrame(result)
        result.columns = ['artist_id', 'ds', 'plays']
        result = result.groupby(['artist_id', 'ds']).sum()
        result = result.reset_index()
        
        result['plays'] = result['plays'].astype(int)
        if self.toDate <= '20150630':
            result[['artist_id', 'ds', 'plays']].to_sql('new_songs_incr', 
                                                        self.mysql_cn, flavor='mysql', 
                                                        if_exists='replace', 
                                                        index = False)
        result.to_csv('%s' %result_file_name, index=False)
         
    def localTest(self):
        cur = self.mysql_cn.cursor()
        genMEScore = '''
        select  artist_id,sqrt(sum(e.target)) * (1-sqrt(avg(pow((e.plays-e.target)/e.target,2)))) as fscore
        from(
            select c.artist_id,c.ds,c.target,d.plays from(
                select t.artist_id,t.ds,t.target
                from (
                    select artist_id, ds, plays as target from
                    artist_play
                    WHERE ds >= '{target_from_date}'
                    and ds <= '{target_to_date}' 
                )t
            )c
            left outer join(
                select artist_id,sum(1/plays)/sum(1/plays/plays) as plays 
                from(
                    select artist_id,ds, plays
                    from artist_play                
                    WHERE ds >= "{me_from_date}"
                    and ds <= "{toDate}"    
                )b group by artist_id
            )d on(c.artist_id=d.artist_id)
        )e group  by artist_id
        order by artist_id;
        '''
        cur.execute(genMEScore.format(target_from_date = self.tfromDate, 
                                      target_to_date = self.ttoDate,
                                      me_from_date = self.me_from_date,
                                      toDate = self.toDate))
        me_result = cur.fetchall()
        
        cur.execute('DROP TABLE IF EXISTS new_songs_tendency;')
#         cur.execute('DROP TABLE IF EXISTS baseline;')
#         genBaseline = '''
#         CREATE TABLE baseline as
#         SELECT k.*, tmp.plays FROM(
#             SELECT artist_id, ds FROM artist_play
#             WHERE ds >= '{target_from_date}' and ds <= '{target_to_date}'
#             GROUP BY artist_id, ds
#         )k LEFT OUTER JOIN(
#             select artist_id, {me_column} as plays
#             from clean_me
#             where ds = '{toDate}'
#         )tmp
#         on k.artist_id = tmp.artist_id;
#         '''
#         cur.execute(genBaseline.format(target_from_date = self.tfromDate, 
#                                        target_to_date = self.ttoDate,
#                                        toDate = self.toDate,
#                                        me_column = 'b_%d' %self.me_day_use))
        genTest = '''
        CREATE table new_songs_tendency as
        SELECT b.artist_id, b.ds,  if ((b.plays + d.plays) is NULL, b.plays,  (b.plays + d.plays))as plays FROM
        baseline b
        LEFT JOIN new_songs_incr d
        on (b.artist_id = d.artist_id and b.ds = d.ds);
        '''
        cur.execute(genTest)
        genScore = '''
        select artist_id,sqrt(sum(e.target)) * (1-sqrt(avg(pow((e.plays-e.target)/e.target,2)))) as fscore
        from(
          select c.artist_id,c.ds,c.target,d.plays from(
              select t.artist_id,t.ds,t.target
              from (
                  select artist_id, ds, plays as target from
                  artist_play
                  WHERE ds >= '{target_from_date}'
                  and ds <= '{target_to_date}' 
              )t
          )c
          left outer join
                  new_songs_tendency d on(c.artist_id=d.artist_id and c.ds = d.ds)
        )e group  by artist_id
        order by artist_id;
        '''
        cur.execute(genScore.format(target_from_date = self.tfromDate, 
                                    target_to_date = self.ttoDate))
        final_me_score = 0.0
        final_pred_score = 0.0
        pred_result = cur.fetchall()
        combine_result = np.hstack((np.array(me_result), np.array(pred_result))).tolist()
        
        for me_artist_id, me_score, pred_artist_id, pred_score in combine_result:
            if self.ifShowEachArtistScore:
                assert me_artist_id == pred_artist_id
                if me_artist_id in self.decay_artist:
                    print me_artist_id, "ME:", me_score, 'Pred:', pred_score
            final_me_score += float(me_score)
            final_pred_score += float(pred_score)
        return (final_me_score, final_pred_score)
if __name__ == '__main__':
    toDate = '20150615'
    nsd = NewSongDecay(toDate=toDate, 
#                     bySelfModel = True,
                       ifDebug=False, 
                       ifPlot=True, 
#                        ifPlotTrain = True,
                       first_use = False, 
                       ifTestOnline = False,
                       ifShowEachArtistScore = True)
#     nsd.setOtherXFeatures({'test':[1,2,3]})
    nsd.getTendency()
    if toDate <= '20150630':
        scores = nsd.localTest()
        print "ME score:", scores[0], "Pred score:", scores[1]
    elif toDate == '20150830':
        print nsd.getOnlineSQL(), nsd.getResultSQL()
