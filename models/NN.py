# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Jun 15, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import numpy
from keras.layers import Input
numpy.random.seed(123)
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn import neighbors
from sklearn.preprocessing import Normalizer

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape
from keras.engine.topology import Merge
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
ss=None

import copy

import pickle
class NN_with_EntityEmbedding(object):

    def __init__(self, X_train, y_train, X_val, y_val):
        self.nb_epoch = 10
        self.others_dim=X_train.shape[1]-4
        print self.others_dim
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)


    def preprocessing(self, x):
        X = copy.deepcopy(x)
        X.fillna(value = 0)
        X["gmt_date"]=pd.to_datetime(X["gmt_date"],format="%Y%m%d")
        gmt_date = X.pop("gmt_date")
        artist_id = X.pop('artist_id')

        week = gmt_date.dt.dayofweek
        days = (gmt_date- min(gmt_date)).astype('timedelta64[D]').astype('int')
        X_list = [artist_id, week, days, X]
        #self.others_dim = X.ndim
        if ss==None:
            ss=StandardScaler()
            ss=ss.fit(X)
            X=ss.transform(X)
        else:
            X=ss.transform(X)
            
        return [x.values for x in X_list]

    def __build_keras_model(self):
        models = []

        model_artist_id = Sequential()
        model_artist_id.add(Embedding(100, 10, input_length=1))
        model_artist_id.add(Reshape(target_shape=(10,)))
        models.append(model_artist_id)

        model_week = Sequential()
        model_week.add(Embedding(7, 2, input_length=1))
        model_week.add(Reshape(target_shape=(6,)))
        models.append(model_week)
        
#         model_gender = Sequential()
#         model_gender.add(Embedding(1, 3, input_length=1))
#         model_gender.add(Reshape(target_shape=(3,)))
#         models.append(model_gender)

        model_day = Sequential()
        model_day.add(Embedding(1, 10, input_length=1))
        model_day.add(Reshape(target_shape=(10,)))
        models.append(model_day)

#         model_language = Sequential()
#         model_language.add(Embedding(1, 3, input_length=1))
#         model_language.add(Reshape(target_shape=(3,)))
#         models.append(model_language)

        model_others = Sequential()
        model_others.add(Reshape((self.others_dim,), input_shape=(self.others_dim,)))
        models.append(model_others)
        
        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dense(100, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(200, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        return val
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return val
        return numpy.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        le=LabelEncoder()
        le.fit(X_train.artist_id)
        X_train.ix[:,"artist_id"]=le.transform(X_train.artist_id)
        X_val.ix[:,"artist_id"]=le.transform(X_val.artist_id)
        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       nb_epoch=self.nb_epoch, batch_size=64,
                       callbacks=[self.checkpointer],
                       )
        self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
    
    def evaluate(self, X_val, y_val):
        assert(min(y_val) > 0)
        guessed_sales = self.guess(X_val)
        relative_err = numpy.absolute((y_val - guessed_sales) / y_val)
        result = numpy.sum(relative_err) / len(y_val)
        return result

if __name__ == '__main__':
    train_ds = pd.read_csv('./p2_artist_train_one_month.csv')
    test_ds = pd.read_csv('./p2_artist_test_one_month.csv')
    submit_ds = pd.read_csv('./p2_artist_submit_one_month.csv')
    
    y_train_all = train_ds.pop('all_target').values
    y_train_clean = train_ds.pop('artist_target').values
    
    y_test_all = test_ds.pop('all_target').values
    y_test_clean = test_ds.pop('artist_target').values
    
    y_submit_all = submit_ds.pop('all_target')
    y_submit_clean = submit_ds.pop('artist_target')
    
    model = NN_with_EntityEmbedding(train_ds, y_train_clean, test_ds, y_test_clean)
    df = pd.DataFrame(model.guess(submit_ds))
    df.to_csv('./submit.csv', header=False, index=False)
    