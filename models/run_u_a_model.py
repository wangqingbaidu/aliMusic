# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on May 16, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import pandas as pd
import pymysql
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import numpy as np
import time, os, copy, re, time, sys, datetime
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class LinearRegression(object):
    """
    solver: 
        ls for least square method, sgd for gridient descent
    """
    def __init__(self,solver="ls",lr=0.2,max_iter=200,bias=False):
        self.solver=solver
        self.coef_=None
        self.bias=bias
        if self.solver=='sgd':
            self.lr=lr
            self.max_iter=max_iter

    def gradient_descent(self,X,y):
        m=len(y)
        for i in xrange(0,self.max_iter):
            pred=X.dot(self.coef_)
            for j in xrange(0,X.shape[1]):
                tmp=X[:,j]
                errors = (pred - y) * tmp#element-wise multi
                self.coef_[j]=self.coef_[j] - self.lr * np.mean(errors)
        return self.coef_

    def fit(self,X,y):
        if self.bias:
            X = np.hstack([X,np.ones((X.shape[0],1))])

        if self.solver=="ls":
            self.coef_=np.linalg.lstsq(X,y)[0]
        else:
            self.coef_=np.zeros(X.shape[1])
            self.coef_=self.gradient_descent(X,y)

    def predict(self,X):
        if self.bias:
            X = np.hstack([X,np.ones((X.shape[0],1))])

        return X.dot(self.coef_)

def get_train_predict(model = None, use_cache = True, 
                      use_artist_features=False, use_user_artist_features = False):
    mysql_cn= pymysql.connect(host='localhost', port=3306,user='root', passwd='111111', db='music')
    append_ua = lambda x : '_ua' if x else ''
    append_uua = lambda x : '_uua' if x else ''
    X_train_filename = './data/X_train_data%s%s.csv' %(append_ua(use_artist_features), 
                                                       append_uua(use_user_artist_features))
    X_test_filename = './data/X_test_data%s%s.csv' %(append_ua(use_artist_features), 
                                                       append_uua(use_user_artist_features))
    model_type = re.split('\.|\'', "%s" %type(model))
    params = re.split("(|)", "%s" %model.get_params())
    print "Training model %s Got!\nModel params are: %s" %(model_type[-2], params[-1])
    
    
    print "Getting X train data"
    X_train = None
    if os.path.exists(X_train_filename) and use_cache:
        X_train = pd.read_csv(X_train_filename)
        X_train = X_train.fillna(value=0).values
    else:
        X_train = pd.read_sql('select * from user_X_train;', con=mysql_cn)
        if use_artist_features:
            print '\tGetting artist X train data...'
            artist_train = pd.read_sql('select * from user_artist_features_train;', 
                                       con=mysql_cn).replace('NULL', value = 0)
            X_train = pd.concat([X_train, artist_train], axis = 1, ignore_index=True)
            
        if use_user_artist_features:
            print '\tGetting user_and_artist X train data...'
            artist_train = pd.read_sql('select * from user_and_artist_features_train;', 
                                       con=mysql_cn).replace('NULL', value = 0)
            X_train = pd.concat([X_train, artist_train], axis = 1, ignore_index=True)
        X_train = X_train.fillna(value=0)
        X_train.to_csv(X_train_filename, header =False, index=False)
        X_train = X_train.astype(float).values
        
    print "Getting y train data"
    y_train = None
    if os.path.exists('./data/y_train_data.csv') and use_cache:
        y_train = pd.read_csv('./data/y_train_data.csv')
        y_train = y_train.fillna(value=0).values
    else:
        y_train = pd.read_sql('select * from user_y_train;', con=mysql_cn)
        y_train = y_train.fillna(value=0)
        y_train.to_csv('./data/y_train_data.csv', header =False, index=False)
        y_train = y_train.values
        
    y_train = y_train.reshape((y_train.shape[0]))
    
    print "Getting X test data"
    X_test = None
    if os.path.exists(X_test_filename) and use_cache:
        X_test = pd.read_csv(X_test_filename)
        X_test= X_test.fillna(value=0).values
    else:    
        X_test= pd.read_sql('select * from user_X_test;', con=mysql_cn)
        if use_artist_features:
            print '\tGetting artist X test data...'
            artist_test = pd.read_sql('select * from user_artist_features_test;', 
                                       con=mysql_cn).replace('NULL', value = 0)
            X_test = pd.concat([X_test, artist_test], axis = 1, ignore_index=True)
        if use_user_artist_features:
            print '\tGetting user_and_artist X test data...'
            artist_test = pd.read_sql('select * from user_and_artist_features_test;', 
                                       con=mysql_cn).replace('NULL', value = 0)
            X_test = pd.concat([X_test, artist_test], axis = 1, ignore_index=True)
        X_test= X_test.fillna(value=0)
        X_test.to_csv(X_test_filename, header =False, index=False) 
        X_test= X_test.astype(float).values
    
    print 'Fitting data...'
    model.fit(X_train, y_train)
    
    print 'Predicting data...'
    y_test = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    
    print "Getting test keys"
    keys_test = None
    if os.path.exists('./data/keys_test_data.csv') and use_cache:
        keys_test = pd.read_csv('./data/keys_test_data.csv')
    else:
        keys_test = pd.read_sql('select * from user_keys_test;', con=mysql_cn)
        keys_test.to_csv('./data/keys_test_data.csv', header =False, index=False)
    
    res = pd.concat([keys_test, y_test], axis = 1, ignore_index=True)
    mysql_cn.close()
    return res

def get_test_predict(model = None, use_cache = True, 
                     use_artist_features = False, use_user_artist_features = False):
    mysql_cn= pymysql.connect(host='localhost', port=3306,user='root', passwd='111111', db='music')
    
    append_ua = lambda x : '_ua' if x else ''
    append_uua = lambda x : '_uua' if x else ''
    X_train_filename = './data/X_test_data%s%s.csv' %(append_ua(use_artist_features), 
                                                       append_uua(use_user_artist_features))
    X_test_filename = './data/X_submit_data%s%s.csv' %(append_ua(use_artist_features), 
                                                       append_uua(use_user_artist_features))
    
    model_type = re.split('\.|\'', "%s" %type(model))
    params = re.split("(|)", "%s" %model.get_params())
    print "Testing model %s Got!\nModel params are: %s" %(model_type[-2], params[-1])
    
    print "Getting X test data"
#     X_train = None
#     if os.path.exists(X_train_filename) and use_cache:
    X_train = pd.read_csv(X_train_filename)
    X_train = X_train.fillna(value=0).values
#     else:
#         X_train = pd.read_sql('select * from user_X_test;', con=mysql_cn)
#         if use_artist_features:
#             print '\tGetting artist X test data...'
#             artist_train = pd.read_sql('select * from user_artist_features_test;', 
#                                        con=mysql_cn).replace('NULL', value = 0)
#             X_train = pd.concat([X_train, artist_train], axis = 1, ignore_index=True)
#         if use_user_artist_features:
#             print '\tGetting user_and_artist X test data...'
#             artist_train = pd.read_sql('select * from user_and_artist_features_test;', 
#                                        con=mysql_cn).replace('NULL', value = 0)
#             X_train = pd.concat([X_train, artist_train], axis = 1, ignore_index=True)
#         X_train = X_train.fillna(value=0)
#         X_train.to_csv(X_train_filename, header =False, index=False)
#         X_train = X_train.astype(float).values
        
    print "Getting y test data"
    y_train = None
    if os.path.exists('./data/y_test_data.csv') and use_cache:
        y_train = pd.read_csv('./data/y_test_data.csv')
        y_train = y_train.fillna(value=0).values
    else:
        y_train = pd.read_sql('select * from user_y_test;', con=mysql_cn)
        y_train = y_train.fillna(value=0)
        y_train.to_csv('./data/y_test_data.csv', header =False, index=False)
        y_train = y_train.values
        
    y_train = y_train.reshape((y_train.shape[0]))
    
    print "Getting X submit data"
    X_test = None
    if os.path.exists(X_test_filename) and use_cache:
        X_test = pd.read_csv(X_test_filename)
        X_test= X_test.fillna(value=0).values
    else:    
        X_test= pd.read_sql('select * from user_X_submit;', con=mysql_cn) 
        if use_artist_features:
            print '\tGetting artist X submit data...'
            artist_test = pd.read_sql('select * from user_artist_features_submit;', 
                                       con=mysql_cn).replace('NULL', value = 0)
            X_test = pd.concat([X_test, artist_test], axis = 1, ignore_index=True)
        if use_user_artist_features:
            print '\tGetting user_and_artist X submit data...'
            artist_test = pd.read_sql('select * from user_and_artist_features_submit;', 
                                       con=mysql_cn).replace('NULL', value = 0)
            X_test = pd.concat([X_test, artist_test], axis = 1, ignore_index=True)
        X_test= X_test.fillna(value=0)
        X_test.to_csv(X_test_filename, header =False, index=False)
        X_test= X_test.astype(float).values
    
    print 'Fitting data...'
    model.fit(X_train, y_train)
    
    print 'Predicting data...'
    y_test = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    
    print "Getting submit keys"
    keys_test = None
    if os.path.exists('./data/keys_submit_data.csv') and use_cache:
        keys_test = pd.read_csv('./data/keys_submit_data.csv')
    else:
        keys_test = pd.read_sql('select * from user_keys_submit;', con=mysql_cn)
        keys_test.to_csv('./data/keys_submit_data.csv', header =False, index=False)
        
    res = pd.concat([keys_test, y_test], axis = 1, ignore_index=True)
    mysql_cn.close()
    return res
 
def gen_predic_csv(dateNow = time.strftime('%Y%m%d'), 
                   timeNow = time.strftime('%H%M%S'), 
                   use_cache = True,
                   use_artist_features = False,
                   use_user_artist_features = False):
#     model = RandomForestRegressor(n_jobs=-1,
#                                 n_estimators=100,
#                                 max_features=5,#5
#                                 max_depth=8,#8
#                                 min_samples_leaf=2,
#                                   random_state=219)
#     model = LinearSVR(C=18,random_state=219)#17
#     model=RandomForestRegressor(
#             n_estimators=100, 
#             random_state=219,
#             n_jobs=-1,
#             min_samples_split=4)#438*
    model=xgb.XGBRegressor(
         max_depth=8, 
         learning_rate=0.03, 
         n_estimators=1000, 
#          silent=True, 
#          objective='count:poisson',#reg:linear,count:poisson 
        nthread=-1, 
#          gamma=0., 
#          min_child_weight=2, 
#          max_delta_step=2, 
#          subsample=0.8, 
#          colsample_bytree=0.3, 
#          colsample_bylevel=1, 
#          reg_alpha=0, 
#          reg_lambda=10, 
        seed=219, 
        missing=None)
    log_file = open('./%s/%s.log' %(dateNow, timeNow), 'a')
    log_file.write(re.split("(|)", "%s" %model.get_params())[-1] + '\n')
    log_file.close()
    test_model = copy.deepcopy(model)
    submit_model = copy.deepcopy(model)
    get_train_predict(test_model, use_cache, use_artist_features, use_user_artist_features).to_csv(
        './%s/%s_train.csv' %(dateNow, timeNow), header=False, index = False)
    
    get_test_predict(submit_model, use_cache, use_artist_features, use_user_artist_features).to_csv(
        './%s/%s_test.csv' %(dateNow, timeNow), header=False, index = False) 
    
# def get_songDic(ifile = None):
#     assert ifile
#     songDic = {}
#     f = open(ifile).readlines()
#     for item in f:
#         items = item.split(',')
#         if not songDic.has_key(items[0]):
#             songDic[items[0]] = items[1]
#     return songDic
#     
# def get_artist(ifile = None, songdic= None):
#     assert ifile and songdic
#     f = open(ifile).readlines()
#     res = []
#     for item in f:
#         items = item.split(',')
#         if songdic.has_key(items[1]):
#             if items[2] == '20151031' or items[2] == '20150831':
#                 continue
#             res.append([songdic[items[1]], float(items[3].split('\n')[0]), items[2]])
#     return res
'''
This function is used to fine tuning the results using lr.
'''
def gen_finally_results(dateNow = None, timeNow = None):
    print '--------------------------Fine tuning model ------------------------------'
    mysql_cn= pymysql.connect(host='localhost', port=3306,user='root', passwd='111111', db='music')
    train_X_data = pd.read_csv('./%s/%s_test_results_avg.csv' %(dateNow, timeNow),
                             names=['artist_id', 'plays', 'ds'])
    train_X_data = train_X_data.sort_values(by=['artist_id', 'ds'])
    train_X_data['plays'] = train_X_data['plays'].astype(int)
#     print train_X_data
    train_y_data = pd.read_sql("""
    SELECT test_3000_lines.artist_id, plays, test_3000_lines.ds FROM
    test_3000_lines 
    LEFT JOIN(
    SELECT artist_id, avg(plays) as plays, ds from(    
        SELECT test_3000_lines.artist_id, plays, test_3000_lines.ds from 
            test_3000_lines 
        LEFT JOIN(
            SELECT artist_id, count(*) as plays, ds from
            user_actions left JOIN songs
            on user_actions.song_id = songs.song_id
            WHERE ds >= '20150702' and ds <= '20150830' and action_type = '1'
            GROUP BY ds, artist_id
            ORDER BY artist_id, ds)a
        on test_3000_lines.artist_id = a.artist_id and test_3000_lines.ds = a.ds
        ORDER BY ds
        LIMIT 50, 3000)c
        GROUP BY artist_id
    )avgtmp
    on test_3000_lines.artist_id = avgtmp.artist_id
    ORDER BY ds
    LIMIT 50, 3000
    """, mysql_cn)
    train_y_data = train_y_data.fillna(value=0)
    train_y_data = train_y_data.sort_values(by=['artist_id', 'ds'])
    train_y_data['plays'] = train_y_data['plays'].astype(int)
#     print train_y_data
#     print train_y_data['plays'].values.shape
    model = LinearRegression()
    model.fit(train_X_data['plays'].values.reshape((train_X_data['plays'].values.shape[0],1)), 
              train_y_data['plays'].values.reshape(train_y_data['plays'].values.shape[0]))
    
    submit_X_data = pd.read_csv('./%s/%s_submit_results.csv' %(dateNow, timeNow),
                             names=['artist_id', 'plays', 'ds'])
    submit_X_data = submit_X_data.sort_values(by=['artist_id', 'ds'])
    submit_X_data['plays'] = submit_X_data['plays'].astype(int)
#     print submit_X_data
    plays = pd.DataFrame(model.predict(
        submit_X_data['plays'].values.reshape((submit_X_data['plays'].values.shape[0],1))))
    submit_X_data['plays'] = plays.astype(int)
    submit_X_data.sort_values(by=['artist_id', 'ds'])
    print 'Saving submit results...'
    submit_X_data.to_csv('./submit/submit_results.csv', header =False, index=False)    
    
    get_avg_results('./submit/submit_results.csv')
    get_min_error_mean_results('./submit/submit_results.csv')
    
#     print 'LR params is ', model.get_params
"""
The following two functions generate the min error results.
"""
def get_min_error_res(play_res):
    res_sum = 0
    res_sum_2 = 0
    for res in play_res:
        res_sum += res
        res_sum_2 += (res*res)
    if res_sum == 0: return 0
    return res_sum_2 / res_sum

def get_min_error_mean_results(in_filename):
    """
    in_filename: artist_id, times, ds
    """
    keys = ['artist_id', 'times', 'ds']
    artist = {}
    data = pd.read_csv(in_filename, header = None, names =  keys)
    days = set()
    for _, row in data.iterrows():
        artist_id = row[keys[0]]
        if artist_id not in artist:
            artist[artist_id] = []
        artist[artist_id].append(row[keys[1]])
        days.add(row[keys[2]])
    days = [day for day in days]
    sorted(days)
    out_filename= in_filename.replace('.csv', '_me.csv')
    results = []
    for artist_id, times in artist.iteritems():
        min_error_res = int(get_min_error_res(times))
        for day in days:
            results.append([artist_id, min_error_res, day])
    df = pd.DataFrame(results)
    df.columns = ['artist_id', 'plays', 'ds']
    df = df.sort_values(by = ['artist_id', 'ds'])
    df.to_csv(out_filename, header =False, index=False)

"""
The following two functions generate the average results.
"""
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

def get_avg_results(ifile_name = None, ratio = 1):
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
                 
def get_f_score(y_true,y_pred):
    '''
    both y_true and y_pred should be 1D array
    '''
    sig=np.sqrt(np.sum(y_true))
    data=[]
    for i,j in zip(y_true,y_pred):
        if i==0:
            continue
        data.append(np.power((j*1.0-i)/i,2))
    #delta=np.sqrt(np.mean(np.power((y_pred-y_true)/y_true,2)))
    delta=np.sqrt(np.mean(np.array(data)))
    #print sig,delta
    return (1-delta)*sig

def evaluate(dateNow = None, timeNow = None):
    print '------------------------Evaluating results---------------------------------'    
    mysql_cn= pymysql.connect(host='localhost', port=3306,user='root', passwd='111111', db='music')
    ifile_name = './%s/%s_test_results.csv' %(dateNow, timeNow)
    for ifile in [ifile_name, 
                 ifile_name.replace('.csv', '_avg.csv'), 
                 ifile_name.replace('.csv', '_me.csv')]:
        Id_date=pd.read_csv("./data/test.csv")[["artist_id","gmt_date","artist_target"]]
        Id_pred=pd.read_csv(ifile, 
                            names=['artist_id', 'ypred', 'gmt_date'])
    
        fscore=0.0
        artists=Id_date.artist_id.unique()
        for _,artist in enumerate(artists):
            df=Id_date[Id_date.artist_id==artist]
            df=df.sort_values(by="gmt_date")
    
            df2=Id_pred[Id_pred.artist_id==artist]
            df2=df2.sort_values(by="gmt_date")
    
            f=get_f_score(df["artist_target"],df2["ypred"])
            #print artist,f
            fscore+=f
            
        y_true = pd.read_sql("""
        SELECT test_3000_lines.artist_id, plays, test_3000_lines.ds from 
        test_3000_lines LEFT JOIN(
        SELECT artist_id, count(*) as plays, ds from
        user_actions left JOIN songs
        on user_actions.song_id = songs.song_id
        WHERE ds >= '20150702' and ds <= '20150830' and action_type = '1'
        GROUP BY ds, artist_id
        ORDER BY artist_id, ds)a
        on test_3000_lines.artist_id = a.artist_id and test_3000_lines.ds = a.ds
        ORDER BY ds
        LIMIT 50, 3000
        """, mysql_cn)
        y_true = y_true.fillna(value=0)
        y_true = y_true.sort_values(by=['artist_id', 'ds'])
        y_true = y_true['plays'].values
        Id_pred = Id_pred.sort_values(by=['artist_id', 'gmt_date'])
        y_pred = Id_pred['ypred'].values
        
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print '#####%s#####' %ifile
        try:
            from termcolor import colored
            print colored('final fscore', 'red'),":",colored(fscore, 'green')
            print colored('r2 score', 'red'),":",colored(r2, 'green')
            print colored('mse', 'red'),":",colored(mse, 'green')
        except:
            print 'final fscore:',fscore
            print 'r2 score:', r2
            print 'mse:', mse
        
        log_file = open('./%s/%s.log' %(dateNow, timeNow), 'a')
        log_file.write('#####%s#####\n' %ifile)
        log_file.write('Final fscore:%f\n' %fscore)
        log_file.write('r2 score:%f\n' %r2)
        log_file.write('mse:%f\n' %mse)
        log_file.close()

def gen_alter_result_csv(dateNow = None, timeNow = None):
    assert dateNow and timeNow
    print '-----------------------Generating alter results-----------------------------'
    file_list = ['./%s/%s_test_results.csv' %(dateNow, timeNow),
                 './%s/%s_submit_results.csv' %(dateNow, timeNow)]
    for ifile_name in file_list: 
        get_avg_results(ifile_name)
        get_min_error_mean_results(ifile_name)
        print ifile_name, ' altered!'

def gen_result_csv(dateNow = time.strftime('%Y%m%d'),
                   timeNow = time.strftime('%H%M%S'),
                   use_cache = True, 
                   del_temp_result = True, 
                   use_artist_features = False,
                   use_user_artist_features = False):
    mysql_cn= pymysql.connect(host='localhost', port=3306,user='root', passwd='111111', db='music')
    print '-------------------------Getting model...---------------------------------'
    gen_predic_csv(dateNow, timeNow, use_cache, 
                   use_artist_features= use_artist_features,
                   use_user_artist_features = use_user_artist_features)
    
    print '-----------------------Getting train results------------------------------'
    print 'Getting dataframe of train data...'
    df = pd.read_csv('./%s/%s_train.csv' %(dateNow, timeNow), 
                     names=['user_id', 'artist_id', 'ds', 'plays'])
    df.pop('user_id')
    df['plays'] = df['plays'].astype(float)
    df = df.groupby(['artist_id', 'ds']).sum()    
    df['plays'] = df['plays'].astype(int)
    df = df.reset_index()
    ds = df.pop('ds')
    df.insert(2, 'ds', ds)
    
    df.columns = ['artist_id', 'plays', 'ds']
    df.to_sql('tmp_test_result', mysql_cn, flavor='mysql', if_exists='replace', 
              index = False)
    
    df = pd.read_sql('''
    SELECT test_3000_lines.artist_id, tmp_test_result.plays, test_3000_lines.ds FROM
    test_3000_lines left join tmp_test_result
    on test_3000_lines.artist_id = tmp_test_result.artist_id and test_3000_lines.ds = tmp_test_result.ds;''', 
        con=mysql_cn)
    df = df.fillna(value=0)
    df['plays'] = df['plays'].astype(int)
    df.sort_values(by='ds')
    df = df.iloc[50:, :]
    df = df.sort_values(by=['artist_id', 'ds'])
    print 'Saving test results...'
    df.to_csv('./%s/%s_test_results.csv' %(dateNow, timeNow), header =False, index=False)
    
    print '-----------------------Getting test results--------------------------------'
    print 'Getting dataframe of test data...'
    df = pd.read_csv('./%s/%s_test.csv' %(dateNow, timeNow), 
                     names=['user_id', 'artist_id', 'ds', 'plays'])
    df.pop('user_id')
    df['plays'] = df['plays'].astype(float)
    df = df.groupby(['artist_id', 'ds']).sum()    
    df['plays'] = df['plays'].astype(int)
    df = df.reset_index()
    ds = df.pop('ds')
    df.insert(2, 'ds', ds)
    
    df.columns = ['artist_id', 'plays', 'ds']
    df.to_sql('tmp_submit_result', mysql_cn, flavor='mysql', if_exists='replace', 
              index = False)
    
    df = pd.read_sql('''
    SELECT submit_3000_lines.artist_id, tmp_submit_result.plays, submit_3000_lines.ds FROM
    submit_3000_lines left join tmp_submit_result
    on submit_3000_lines.artist_id = tmp_submit_result.artist_id and submit_3000_lines.ds = tmp_submit_result.ds;''', 
        con=mysql_cn)
    df = df.fillna(value=0)
    df['plays'] = df['plays'].astype(int)
    df = df.sort_values(by=['artist_id', 'ds'])
    print 'Saving submit results...'
    df.to_csv('./%s/%s_submit_results.csv' %(dateNow, timeNow), header =False, index=False)
    
    gen_alter_result_csv(dateNow, timeNow)
    gen_finally_results(dateNow, timeNow)
    
    if del_temp_result:
        print '-----------------------Deleting temp results--------------------------------'
        os.system('rm ./%s/%s_train.csv' %(dateNow, timeNow))
        print './%s/%s_train.csv deleted!' %(dateNow, timeNow)
        os.system('rm ./%s/%s_test.csv' %(dateNow, timeNow))
        print './%s/%s_test.csv deleted!' %(dateNow, timeNow)
        
    evaluate(dateNow, timeNow)
    
if __name__ == '__main__':
    print '-------------------------System info--------------------------------------'
    dateNow = time.strftime('%Y%m%d')
    timeNow = time.strftime('%H%M%S')
    dataDirs = ['data', 'submit', 'ago', dateNow]
    for dataDir in dataDirs:
        if not os.path.exists(dataDir):
            os.system('mkdir %s' %dataDir)
    gapday = datetime.timedelta(days=1)
    tomorrow = (datetime.datetime.now() - gapday).strftime('%Y%m%d')
    if os.path.exists('./%s'%tomorrow):
        print 'Moving tomorrow %s files' %tomorrow
        os.system('mv ./%s ./ago' %tomorrow)
    
    args = {'uc':True, 'dt':True, 'ua':True, 'uua':True,
            '-uc':False, '-dt':False, '-ua':False, '-uua':False}
    use_cache = True
    del_temp_result = True
    use_artist_features = True
    use_user_artist_features = True
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            assert args.has_key(arg)
            if arg =='uc' or arg == '-uc':
                use_cache = args[arg]
            elif arg =='dt' or arg == '-dt':
                del_temp_result = args[arg]
            elif arg =='ua' or arg == '-ua':
                use_artist_features = args[arg]
            elif arg =='uua' or arg == '-uua':
                use_user_artist_features = args[arg]
    get_color = lambda x: 'green' if x else 'red'
    
    try:
        from termcolor import colored
        print colored('Use data cache', 'blue'),":",colored(use_cache, get_color(use_cache))
        print colored('Delete temp files', 'blue'),":",colored(del_temp_result, get_color(del_temp_result))
        print colored('Use artist features', 'blue'),":",colored(use_artist_features, get_color(use_artist_features))
        print colored('Use user and artist features', 'blue'),":",colored(use_user_artist_features, get_color(use_user_artist_features))
    except:        
        print 'Use data cache:', use_cache
        print 'Delete temp files:', del_temp_result
        print 'Use artist features:',use_artist_features
        print 'Use user_artist features:',use_user_artist_features
 
    log_file = open('./%s/%s.log' %(dateNow, timeNow), 'a')
    log_file.write('Use data cache:%s\n'%use_cache + 
                   'Delete temp files:%s\n'%del_temp_result + 
                   'Use artist features:%s\n'%use_artist_features + 
                   'Use user_artist features:%s\n'%use_user_artist_features)
    log_file.close()
    timeStart = time.time()
    gen_result_csv(dateNow = dateNow,
                   timeNow = timeNow,
                   use_cache= use_cache, 
                   del_temp_result=del_temp_result, 
                   use_artist_features=use_artist_features,
                   use_user_artist_features=use_user_artist_features)
    
    timeUsed = int(time.time() - timeStart)
    info = '\nTotal use %d min(s) %d sec(s).' %(timeUsed / 60, timeUsed % 60)
    try:
        from termcolor import colored
        print colored(info, 'yellow')
    except:
        print info
#     gen_finally_results('20160528', '144748')
