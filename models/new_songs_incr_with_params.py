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
import random
random.seed(219)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

data2use = {True:'p2_clean_user_actions_with_artists', False:'p2_user_actions_with_artists'}
maxt = {True:None, False:'smaller than maxt'}
lent = {True:None, False:'smaller than lent'}
train_ds = ['20150630', '20150530', '20150430', '20150330']
                
class NewSongDecay:
    """
    Parameters 
    -------------
    @toDate: Must be set, eg. 20150630, it will predict 20150702-20150830
    @bySelfModel: True or False, if False will ignore addInfinity and infinityNumberRatio
    @addInfinity: True or False, use infinite or not
    @infinityNumberRatio: float, 使用的无穷远点个数占总长度的比例
    @updateFactor: True or False, update decay factor or not
    @useFactor: True or False, use decay factor or not
    @ifShowEachArtistScore: True or False, show score of each artist predicted
    
    @max_value_threshold: If the max number of a train set smaller than this it will be ignored 
    @train_length_threshold: If the length of a train set smaller than this it will be ignored
    @r2_threashold: If the prediction of a test set smaller than this it will be treated as unpredictable
    @predict_length_threshold: If the length of a test is smaller than this it will be treated as unpredictable
    """
    def __init__(self, 
                 toDate = None, 
                 bySelfModel = False,
                 addInfinity = False,
                 infinityNumberRatio = .5,
                 new_song_day_use = 61, 
                 me_day_use = 14, 
                 max_value_threshold = 1000, 
                 train_length_threshold = 30,
                 r2_threashold = .5,
                 predict_length_threshold = 10,
                 use_clean = True,
                 local_data2use = 'artist_play',
                 updateFactor = False,
                 useFactor = True,
                 first_use = False,
                 ifTestOnline = False,
                 ifDebug = True,
                 ifPlot = True,
                 ifPlotTrain = False,
                 ifShowEachArtistScore = False,
                 host='10.25.0.118', port=3306,user='root', passwd='111111', db='alimusic'):
        assert toDate
        self.ifDebug = ifDebug
        self.ifTestOnline = ifTestOnline
        self.infinityNumberRatio = infinityNumberRatio
        self.ifShowEachArtistScore = ifShowEachArtistScore
        self.ifPlot = ifPlot
        self.ifPlotTrain = ifPlotTrain,
        self.toDate = toDate
        self.updateFactor = updateFactor
        self.useFactor = useFactor
        self.bySelfModel = bySelfModel
        self.addInfinity = addInfinity
        self.unpredictable = [] 
        self.max_value_threshold = max_value_threshold
        self.train_length_threshold = train_length_threshold
        self.r2_threashold = r2_threashold
        self.predict_length_threshold = predict_length_threshold
        
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
        self.local_data2use = local_data2use
        self.first_use = first_use
        
        self.decay_artist = []
        self.X_additional_features = None
        self.mysql_cn= pymysql.connect(host=host, port=port,user=user, passwd=passwd, db=db)
    
    def __del__(self):
        if self.mysql_cn:
            self.mysql_cn.close()
            
    def __transform_X(self, x):
        return 1.0/(1+np.log2(x)/np.log2(100))
#         return 1.0 / x
     
    def __transform_y(self, y, max_y):
        return y / max_y
     
    def __inverse_transform_y(self, y, max_y):
        return y * max_y
            
    def getOnlineSQL(self):
        return self.genDependence() + self.genPeriodME() + self.genNewSongOutBaseline()
    
    def getResultSQL(self):
        return self.genResultTable()
    
    def genDependence(self):
        sqlTemplate = '''
        drop table if exists zzw_all_keys;
        create table zzw_all_keys as
        select c.artist_id,to_char(dateadd(to_date(c.ds,"yyyymmdd"),61,"dd"),"yyyymmdd") as ds from (
            select artist_id,ds from {data2use}
            where ds>="20150301" and ds<"20150831" group  by artist_id,ds
        )c;
        
        drop table if exists zzw_new_songs;
        create table zzw_new_songs as 
          select a.artist_id, a.ds, a.action_type, b.publish_time from 
              {data2use} a
          join (
              select song_id, publish_time from songs 
              where publish_time >= '20150301' and publish_time <= '20150830'
          )b
          on a.song_id = b.song_id;
          
        drop table if exists new_songs_plays;
        create table new_songs_plays as
          select concat(artist_id, publish_time) as album, ds, count(1) as plays from zzw_new_songs
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
        select artist_id,sum(1/plays)/sum(1/plays/plays) as plays from(
            select song_out.artist_id,song_out.ds,count(1) as plays
            from (
                select fi.* from(
                    select cua.*, s.album from 
                    p2_user_song_with_user_clean_2 cua
                    left outer join(
                        select song_id, concat(artist_id, publish_time) as album from songs
                    )s on s.song_id = cua.song_id
                )fi
                where fi.album not in(
                select album from predictable
                )
            )song_out
            where song_out.action_type=1                        
                and song_out.ds >= "{me_from_date}"
                and song_out.ds <= "{toDate}"    
            group by song_out.artist_id,song_out.ds
        )b 
        group by artist_id;
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
        join
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
            LEFT　OUTER JOIN new_songs_incr d
            on (b.artist_id = d.artist_id and b.ds = d.ds)
            where d.plays > 0;
        '''
        sqlTemplate = '''        
        select sum(f.fscore) as score from(
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
            join
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
    
    def __get_decay_ratio(self, album=None):
        assert album
        decay_factor = self.getDecayFactor()
        if decay_factor and self.useFactor:
            if decay_factor.has_key(album[:-8]):
                if self.ifDebug:
                    print album[:-8], 'Use self factor'
                return decay_factor[album[:-8]]
            else:
                if self.ifDebug:
                    print album[:-8], 'Use global factor'
                return decay_factor[self.toDate]
        else:
            return 0.1
        
    def __gen_data_set(self):
        album_list = [(x[0], x[1]) 
                      for x in pd.read_sql('select album, plays from album_list order by album desc', 
                                           self.mysql_cn).values.tolist()]
        ##add by nanhai.yang
#         albums=[a for a,_ in album_list]
#         X_FEATURES=dict()
#         artists=dict()
#         for album in albums:
#             artists[album[:-8]]=1
#         for i,artist in enumerate(artists):
#             artists[artist]=i
# 
#         for i,album in enumerate(albums):
#             ARTIST_FEATURE=[0]*len(artists)
#             ARTIST_FEATURE[artists[album[0:-8]]]=1
#             X_FEATURES[album]=[1]#+ARTIST_FEATURE
        
#         artists = [album[:-8] for album, _ in album_list]
#         artists = set(artists)
        addition_features = {album : [1] for album, _ in album_list}
        
        self.setOtherXFeatures(addition_features)
        ###################
        X_train = None
        y_train  = None
        first = False 
        test = []
        new_decay_factor = []
        for album, _ in album_list:            
            X = None
            y = None            
#           Use start parameter because of some songs are listened ahead of artist publish time   
            if album[-8:] >= self.new_song_from_date and album[-8:] <= self.toDate:
                df = pd.read_sql('''
                SELECT plays from new_songs_plays
                WHERE album = '{album}' and ds >= '{start}' and ds <= '{end}'
                order by ds;
                '''.format(album=album, start = album[-8:], end = self.toDate),self.mysql_cn)
                y = df.astype(float).values
                try:
                    y_index = y.argmax()
                except:
                    self.unpredictable.append(album)
                    continue 
                y = y[y_index:]
                if y.shape[0] < 5 or y.shape[0] > 45:
                    self.unpredictable.append(album)
                    continue 
                max_y = np.max(y)
                min_y = np.min(y)
                X = [i + 1 for i in range(y.shape[0])]
                if self.addInfinity:
                    infinate_num = int(y.shape[0] * self.infinityNumberRatio)
                    y = np.vstack((y, np.array([max_y * self.__get_decay_ratio(album)] * infinate_num).reshape((-1, 1))))
                    X += [random.randint(60,244) for _ in range(infinate_num)]
                y = self.__transform_y(y, max_y).reshape((-1))
                X = self.__transform_X(np.array(X).reshape((-1, 1)))
#               If additional X features is set, then It must contains all album features 
#                 assert (not self.X_additional_features or 
#                     (self.X_additional_features and self.X_additional_features.has_key(album)))
#                 if self.X_additional_features and self.X_additional_features.has_key(album):
#                     X = np.hstack((X, np.array([self.X_additional_features[album] for _ in range(X.shape[0])])))
                test.append((X, y, album, max_y, y_index))
            else:
                self.unpredictable.append(album)
                df = pd.read_sql('''
                SELECT plays from new_songs_plays
                WHERE album = '{album}' and ds >= '{start}'
                order by ds;
                '''.format(album=album, start = album[-8:]),self.mysql_cn)
                if album[-8:] >= '20150831':
                    continue
                y = df.astype(float).values
                y_index = y.argmax()
                y = y[y_index:]
                max_y = np.max(y)
                min_y = np.min(y)
                new_decay_factor.append([album[:-8], min_y / max_y])
                if max_y < self.max_value_threshold or y.shape[0] < self.train_length_threshold:
                    if self.ifDebug:
                        info = [maxt[y.shape[0] >= self.train_length_threshold], 
                                lent[max_y >= self.max_value_threshold]]
                        print "Album %s dropped beacause of %s" %(album, ' and '.join([s for s in info if s]))
                    continue 
                X = [i + 1 for i in range(y.shape[0])]
                if self.addInfinity:
                    infinate_num = int(y.shape[0] * self.infinityNumberRatio)
                    y = np.vstack((y, np.array([max_y * self.__get_decay_ratio(album)] * infinate_num).reshape((-1, 1))))
                    X += [random.randint(60,365) for _ in range(infinate_num)]
                y = self.__transform_y(y, max_y).reshape((-1))
                X = self.__transform_X(np.array(X).reshape((-1, 1)))
#               If additional X features is set, then It must contains all album features 
                assert (not self.X_additional_features or 
                    (self.X_additional_features and self.X_additional_features.has_key(album)))
                if self.X_additional_features and self.X_additional_features.has_key(album):
                    X = np.hstack((X, np.array([self.X_additional_features[album] for _ in range(X.shape[0])])))
                if not first:
                    y_train = y
                    X_train = X
                    first = True
                else:
                    y_train = np.hstack((y_train, y))
                    X_train = np.vstack((X_train, X))

        if self.updateFactor:
            new_decay_factor = pd.DataFrame(new_decay_factor)
            new_decay_factor.columns = ['artist_id', 'factor']
            new_decay_factor['factor'] = new_decay_factor['factor'].astype(float)
            new_decay_factor = new_decay_factor.groupby(['artist_id']).mean().reset_index()
            mean_factor = new_decay_factor['factor'].mean()
            new_decay_factor = new_decay_factor.append(pd.DataFrame([[self.toDate, mean_factor]], 
                                                                    columns=['artist_id', 'factor']))
            new_decay_factor.to_csv('decay_factor.csv', header =False, index=False)
        self.data_set = (X_train, y_train, test)
    
    def getDataSet(self):
        try:
            return self.data_set
        except:
            self.__gen_data_set()
            return self.data_set
    
    def __gen_decay_factor_dict(self):         
        decay_factor = pd.read_csv('decay_factor.csv', names = ['artist_id', 'factor'])
#         print decay_factor.iloc[decay_factor.shape[0] - 1, 0]
        if decay_factor.iloc[decay_factor.shape[0] - 1, 0] != self.toDate:
            self.decay_factor = None
        else:
            self.decay_factor = {x[0]:float(x[1]) for x in decay_factor.values.tolist()}
        
    def getDecayFactor(self):
        try:
            return self.decay_factor
        except:
            self.__gen_decay_factor_dict()
            return self.decay_factor

    #advised by nanhai.yang
    def __gen_model(self, model = LinearRegression(fit_intercept=False)):
        X_train, y_train, _ = self.getDataSet()
        model.fit(X_train, y_train)
        #add by nanhai.yang
        if self.ifPlotTrain:
            y_pred = model.predict(X_train)
            df = pd.DataFrame(np.hstack((y_train.reshape(-1,1), y_pred.reshape(-1,1))))
            df.columns = ['Train', 'Predict']
            df[:60].plot()
            plt.title('train_all')
            fig = plt.gcf()
            fig.savefig('./img/train_all.png')
            plt.close(fig)
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
        predictable = []
        for xt, yt, album, max_y, index in test:
            gapday = datetime.timedelta(days=int(index))
            dateFrom = datetime.datetime.strptime(album[-8:], '%Y%m%d')
            X = self.__transform_X(X_all)
            self_X = xt
            if self.X_additional_features and self.X_additional_features.has_key(album):
                X = np.hstack((X, np.array([self.X_additional_features[album] for _ in range(X.shape[0])])))
                self_X = np.hstack((xt, np.array([self.X_additional_features[album] for _ in range(xt.shape[0])])))
            model = None
            if self.bySelfModel:
                model = LinearRegression(fit_intercept=False)
                model.fit(self_X, yt)
                print album, model.coef_
            else:
                model = self.getModel()
            pred = model.predict(X).reshape((-1,1))
            pred = self.__inverse_transform_y(pred, max_y).reshape((-1))
            if self.ifDebug:
                print album, index
                
            if not album[:-8] in self.decay_artist:
                self.decay_artist.append(album[:-8])
                
#           Use r2 and MSE to envaluate the predictions
            r2 = r2_score(yt, self.__transform_y(pred[:yt.shape[0]], max_y))
            if r2 >= self.r2_threashold and yt.shape[0] > self.predict_length_threshold:
                predictable.append(album)
            else:
                self.unpredictable.append(album)
                continue
            mse = mean_squared_error(yt, self.__transform_y(pred[:yt.shape[0]], max_y)) * 10000
            
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
                        df.plot()
                        plt.title('%s-%.2f r2:%.2f mse:%.2f' %(album[:8], plays / max_y * 100, r2, mse))
                        fig = plt.gcf()
                        fig.set_size_inches(16, 9)
                        fig.savefig('./img/{toDate}_{album}.png'.format(album = album, toDate = self.toDate),
                                    dpi = 100)
                        plt.close(fig)
                    break
                if dateNow <= self.toDate or dateNow > self.ttoDate:
                    continue
                result.append([album[:-8], dateNow, float(plays)])
        pd.DataFrame(np.array(predictable).reshape((-1,1))).to_csv('predictable.csv', 
                                                                   index=False, 
                                                                   header=False)
        
        pd.DataFrame(np.array(predictable).reshape((-1,1)), 
                     columns=['album']).to_sql('predictable',
                                               self.mysql_cn, flavor='mysql', 
                                               if_exists='replace', 
                                               index = False)
        pd.DataFrame(np.array(self.unpredictable).reshape((-1,1)), 
                     columns=['album']).to_sql('unpredictable',
                                               self.mysql_cn, flavor='mysql', 
                                               if_exists='replace', 
                                               index = False)
        if not result:
            result.append(['NULL', 'NULL', 0])
        result = pd.DataFrame(result, columns = ['artist_id', 'ds', 'plays'])
        result = result.groupby(['artist_id', 'ds']).sum()
        result = result.reset_index()
        result['plays'] = result['plays'].astype(int)
        if self.toDate <= '20150630':
            result[['artist_id', 'ds', 'plays']].to_sql('new_songs_incr', 
                                                        self.mysql_cn, flavor='mysql', 
                                                        if_exists='replace', 
                                                        index = False)
            cur = self.mysql_cn.cursor()        
            cur.execute('CREATE INDEX IDX_nsi ON new_songs_incr(artist_id, ds);')
        result.to_csv('%s' %result_file_name, index=False)
         
    def localTest(self):
        cur = self.mysql_cn.cursor()        
        genMEScore = '''
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
            left outer join(
                select artist_id,sum(1/plays)/sum(1/plays/plays) as plays 
                from(
                    select artist_id,ds, plays
                    from {local_data2use}                
                    WHERE ds >= "{me_from_date}"
                    and ds <= "{toDate}"    
                )b group by artist_id
            )d on(c.artist_id=d.artist_id)
        )e group  by artist_id
        order by artist_id;
        '''
        cur.execute(genMEScore.format(local_data2use = self.local_data2use,
                                      target_from_date = self.tfromDate, 
                                      target_to_date = self.ttoDate,
                                      me_from_date = self.me_from_date,
                                      toDate = self.toDate))
        me_result = cur.fetchall()
#         cur.execute('DROP TABLE IF EXISTS remove_new_songs_plays;')
#         cur.execute('''
#         CREATE TABLE remove_new_songs_plays as
#         SELECT artist_id, ds, SUM(plays) as plays FROM(
#             SELECT SUBSTRING(album,1, CHAR_LENGTH(album) - 8) as artist_id, ds, plays FROM new_songs_plays
#             WHERE album IN (
#                 SELECT album from predictable
#             )
#         )a
#         GROUP BY a.artist_id, a.ds
#         ''')
#         cur.execute('ALTER TABLE remove_new_songs_plays MODIFY COLUMN artist_id nvarchar(48);')
#         cur.execute('CREATE INDEX IDX_rnsp ON remove_new_songs_plays(artist_id, ds);')
#         
#         cur.execute('drop table if exists baseline_temp;')
#         cur.execute('''
#         create table baseline_temp as
#         select artist_id,sum(1/plays)/sum(1/plays/plays) as plays 
#         from(
#             select artist_id,ds, plays
#             from (
#                 SELECT b.artist_id, b.ds, if((b.plays - d.plays) is NULL, b.plays, (b.plays - d.plays))as plays FROM
#                 {local_data2use}  b
#                 left outer JOIN (
#                     SELECT artist_id, ds, SUM(plays) as plays FROM(
#                         SELECT SUBSTRING(album,1, CHAR_LENGTH(album) - 8) as artist_id, ds, plays FROM new_songs_plays
#                         WHERE album IN (
#                             SELECT album from predictable
#                         )
#                     )a
#                     GROUP BY a.artist_id, a.ds
#                 ) d
#                 on (b.artist_id = d.artist_id and b.ds = d.ds)
#             )nap                
#             WHERE nap.ds >= "{me_from_date}"
#             and nap.ds <= "{toDate}"    
#         )b group by artist_id
#         order by artist_id
#         '''.format(me_from_date = self.me_from_date, toDate = self.toDate))
#         cur.execute('CREATE INDEX IDX_bl ON baseline_temp(artist_id);')
#         
#         cur.execute('DROP TABLE IF EXISTS baseline;')
#         genTestBaseline = '''
#         CREATE TABLE baseline as
#         SELECT k.*, tmp.plays FROM(
#             SELECT artist_id, ds FROM {local_data2use}
#             WHERE ds >= '{target_from_date}' and ds <= '{target_to_date}'
#             GROUP BY artist_id, ds
#         )k LEFT OUTER JOIN(
#             select artist_id,sum(1/plays)/sum(1/plays/plays) as plays 
#             from(
#                 select artist_id,ds, plays
#                 from (
#                     SELECT b.artist_id, b.ds, if((b.plays - d.plays) is NULL, b.plays, (b.plays - d.plays))as plays FROM
#                     {local_data2use}  b
#                     left outer JOIN (
#                         SELECT artist_id, ds, SUM(plays) as plays FROM(
#                             SELECT SUBSTRING(album,1, CHAR_LENGTH(album) - 8) as artist_id, ds, plays FROM new_songs_plays
#                             WHERE album IN (
#                                 SELECT album from predictable
#                             )
#                         )a
#                         GROUP BY a.artist_id, a.ds
#                     ) d
#                     on (b.artist_id = d.artist_id and b.ds = d.ds)
#                 )nap                
#                 WHERE nap.ds >= "{me_from_date}"
#                 and nap.ds <= "{toDate}"    
#             )b group by artist_id
#             order by artist_id
#         )tmp
#         on k.artist_id = tmp.artist_id;
#         '''
#         cur.execute(genTestBaseline.format(target_from_date = self.tfromDate, 
#                                            target_to_date = self.ttoDate,
#                                            me_from_date = self.me_from_date,
#                                            toDate = self.toDate))
#         
        cur.execute('DROP TABLE IF EXISTS new_songs_tendency;')
        genTest = '''
        CREATE table new_songs_tendency as
        SELECT b.artist_id, b.ds,  if ((b.plays + d.plays) is NULL, b.plays,  (b.plays + d.plays))as plays FROM(
            SELECT k.*, tmp.plays FROM(
                SELECT artist_id, ds FROM {local_data2use}
                WHERE ds >= '{target_from_date}' and ds <= '{target_to_date}'
                GROUP BY artist_id, ds
            )k LEFT OUTER JOIN(
                select artist_id,sum(1/plays)/sum(1/plays/plays) as plays 
                from(
                    select artist_id,ds, plays
                    from (
                        SELECT b.artist_id, b.ds, if((b.plays - d.plays) is NULL, b.plays, (b.plays - d.plays))as plays FROM
                        {local_data2use}  b
                        left outer JOIN (
                            SELECT artist_id, ds, SUM(plays) as plays FROM(
                                SELECT SUBSTRING(album,1, CHAR_LENGTH(album) - 8) as artist_id, ds, plays FROM new_songs_plays
                                WHERE album IN (
                                    SELECT album from predictable
                                )
                            )a
                            GROUP BY a.artist_id, a.ds
                        ) d
                        on (b.artist_id = d.artist_id and b.ds = d.ds)
                    )nap                
                    WHERE nap.ds >= "{me_from_date}"
                    and nap.ds <= "{toDate}"    
                )b group by artist_id
                order by artist_id
            )tmp
            on k.artist_id = tmp.artist_id
        )b 
        LEFT JOIN new_songs_incr d
        on (b.artist_id = d.artist_id and b.ds = d.ds)
        order by b.artist_id, b.ds;
        '''
        cur.execute(genTest.format(local_data2use = self.local_data2use,
                                   target_from_date = self.tfromDate, 
                                   target_to_date = self.ttoDate,
                                   me_from_date = self.me_from_date,
                                   toDate = self.toDate))
        
        cur.execute('CREATE INDEX IDX_NST ON new_songs_tendency(artist_id, ds);')
        
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
          join
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
        me_pred = []
        for me_artist_id, me_score, pred_artist_id, pred_score in combine_result:
            if self.ifShowEachArtistScore:
                assert me_artist_id == pred_artist_id
                if me_artist_id in self.decay_artist:
                    if self.ifDebug:
                        print me_artist_id, "ME:", me_score, 'Pred:', pred_score
                    me_pred.append([me_artist_id, me_score, pred_score])
            final_me_score += float(me_score)
            final_pred_score += float(pred_score)
        if self.ifShowEachArtistScore:
            pd.DataFrame(me_pred).to_csv('each_artist_score.csv', header =['artist_id', 'ME', 'Pred'], index=False)
        return (final_me_score, final_pred_score)
    
def local_test_all(bySelfModel = False, addInfinity = False):
    s = '20150630'
    res = []
    for days in range(50):
        toDate = (datetime.datetime.strptime(s, '%Y%m%d') - datetime.timedelta(days=days)).strftime('%Y%m%d')
        for r in range(6):
            r2t = (r + 3) / 10.0
            for p in range(5):
                nsd = NewSongDecay(toDate=toDate, 
                               local_data2use = 'clean_artist_play',
                               me_day_use = 9,
                               bySelfModel = bySelfModel,
                               addInfinity = addInfinity,
                               infinityNumberRatio = .3,
                               updateFactor = True,
                               r2_threashold = r2t,
                               predict_length_threshold = p + 6,
                               ifTestOnline = False,
                               ifDebug=False,
                               ifPlot=True, 
                               ifShowEachArtistScore = True
                               )
                nsd.getTendency()
                scores = nsd.localTest()
                res.append(['%.2f_%d' %(r2t, p), scores[0], scores[1], scores[1] / scores[0]])
                print '---------%s----%.2f----%d-------' %(toDate, r2t, p)
                print "ME score:", scores[0], "Pred score:", scores[1]
    pd.DataFrame(res).to_csv('total_res.csv', header =['Params', 'ME', 'Pred', 'Promote'], index=False)
if __name__ == '__main__':
#     local_test_all(bySelfModel = True, addInfinity = True)
#     exit()
    for toDate in ['20150630']:
        nsd = NewSongDecay(toDate=toDate, 
                           local_data2use = 'clean_artist_play',
                           me_day_use = 9,
                           bySelfModel = True,
                           addInfinity = True,
                           infinityNumberRatio = .3,
                           updateFactor = True,
                           r2_threashold = .3,
                           predict_length_threshold = 8,
    #                        max_value_threshold = 0, 
    #                        train_length_threshold = 0,
    #                        ifPlotTrain = True,
    #                        first_use = True, 
                           ifTestOnline = False,
                           ifDebug=False,
                           ifPlot=True, 
                          ifShowEachArtistScore = True
                           )
    #     nsd.setOtherXFeatures({'test':[1,2,3]})
        nsd.getTendency()
        if toDate <= '20150630':
            scores = nsd.localTest()
            print nsd.tfromDate, '-->', nsd.ttoDate
            print "ME score:", scores[0], "Pred score:", scores[1]
        elif toDate == '20150830':
            print nsd.getOnlineSQL(), nsd.getResultSQL()
