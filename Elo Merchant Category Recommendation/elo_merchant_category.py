# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:25:09 2019

@author: Y435
"""
import gc
import load as ld
import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
import training as tr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

def baseline_nn(n_features=75):
    model = Sequential()
    model.add(Dense(64, input_dim=n_features, init='normal', activation='sigmoid'))
    model.add(Dense(128, init='normal', activation='sigmoid'))
    model.add(Dense(1, init='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def get_tree_models():
    lgb_params = {}
    #lgb_params['nthread'] = 3
    lgb_params['n_estimators'] = 10000
    lgb_params['learning_rate'] = 0.005
    lgb_params['colsample_bytree'] = 0.7
    lgb_params['subsample'] = 0.7
    lgb_params['max_depth'] = 9
    #lgb_params["reg_alpha"] = 0.041545473
    #lgb_params['reg_lambda'] = 0.0735294
    #lgb_params['num_leaves'] = 34
    lgb_params['metric'] = 'rmse'
    lgb_params['objective'] = 'regression'
    
    
#    lgb_params = {'num_leaves': 111,
#         'min_data_in_leaf': 149, 
#         'objective':'regression',
#         'max_depth': 9,
#         'learning_rate': 0.005,
#         "boosting": "gbdt",
#         "feature_fraction": 0.7522,
#         "bagging_freq": 1,
#         "bagging_fraction": 0.7083 ,
#         "bagging_seed": 11,
#         "metric": 'rmse',
#         "lambda_l1": 0.2634,
#         "random_state": 133,
#         "verbosity": -1}
    
    lgb_fit_params = {}
    lgb_fit_params['verbose_eval'] = 50
    lgb_fit_params['early_stopping_rounds'] = 200
    lgb_fit_params['valid_sets'] = {}
    lgb_fit_params['valid_names'] = ["validation"]
    
    xgb_params = dict()
    xgb_params["booster"] = "gbtree"
    xgb_params["objective"] = "reg:linear"
    #xgb_params["colsample_bytree"] = 0.9497036
    #xgb_params["subsample"] = 0.8715623
    #xgb_params["max_depth"] = 8
    #xgb_params['reg_alpha'] = 0.041545473
    #xgb_params['reg_lambda'] = 0.0735294
    xgb_params["learning_rate"] = 0.005
    #xgb_params["min_child_weight"] = 39.3259775
    xgb_params['eval_metric'] = 'rmse'
    xgb_params['silent'] = 1
    
    xgb_fit_params = {}
    xgb_fit_params['verbose_eval'] = 100
    xgb_fit_params['early_stopping_rounds'] = 200
    xgb_fit_params['evals'] = {}
    xgb_fit_params['num_boost_round'] = 10000

    tree_models = []   

    lgbm = tr.LightGBMRegressorWrapper(params = lgb_params, name = "lgbm")
    xgb = tr.XgbRegressorWrapper(params = xgb_params, name = "xgb")
    
    tree_models.append((lgbm, lgb_fit_params))
    #tree_models.append((xgb, xgb_fit_params))
        
    return tree_models

def run_lgb(train_X, train_y, val_X, val_y, test_X, features = None, verbose = 50, early_stopping_rounds = 200):
    
    lgb_params = {
            "objective" : "regression",
            "metric" : "rmse",
            "num_leaves" : 30,
            "min_child_weight" : 50,
            "learning_rate" : 0.01,
            "bagging_fraction" : 0.8,
            "feature_fraction" : 0.8,
            "bagging_frequency" : 5,
            "bagging_seed" : 2018,
            "num_iterations" : 2000
        }
       
    if features == None:
        features = train_X.columns.tolist()
        
    #train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 2018)
    
    lgb_train = lgb.Dataset(data = train_X, label = train_y, feature_name = features)
    lgb_val = lgb.Dataset(data = val_X, label = val_y, feature_name = features)
    
    lgb_booster = lgb.train(params = lgb_params, train_set = lgb_train, valid_sets = [lgb_val], valid_names = ["validation"], 
            verbose_eval = verbose, early_stopping_rounds = early_stopping_rounds)
    
    predictions = lgb_booster.predict(test_X, num_iteration = lgb_booster.best_iteration)
    
    return lgb_booster, predictions

def get_datasets(debug_size, silent, treat_duplicated = True):
    train, test = ld.get_processed_files(debug_size, silent)
    features = [f for f in train.columns if f not in ['target', 'card_id', 'index', 'first_active_month']]
    
    train_y = train['target']
    train_X = train.loc[:, features]

    ids = test['card_id']
    test_X = test.loc[:, features]
    
    return train_X, train_y, test_X, ids

if __name__ == "__main__":
    debug_size = 10000
    silent = False
    verbose = 10
    early_stopping_rounds = 100
    
    with pp.timer("Full Model Run"):
        train_X, train_y, test_X, ids = get_datasets(debug_size = debug_size, silent = silent)
        features = train_X.columns.tolist()
        
        print("Regrerssors will be fitted with {} out of {} features".format(len(features), train_X.shape[1]))
        
        for m, fp in get_tree_models():
        
            with pp.timer("Run " + m.name):
                
                model = tr.OOFRegressor(reg = m, nfolds = 5, stratified = False)
                
                model.fit(train_X.loc[:, features], train_y, **fp)
                pred = model.predict(test_X.loc[:, features])
                
                cv_score = model.rmse_score_
                feat_importance = model.importances_
                
                if debug_size == 0:
                    submission = pp.submit_file(ids, pred, prefix_file_name = m.name, cv_score = cv_score)
                
                tr.display_importances(feat_importance, 10)
                
                del model, pred, cv_score, feat_importance
                gc.collect()
                
                print("*" * 80)
        
        sk_params = {"n_features": train_X.shape[1], "batch_size": 32, "verbose" : 1}
        
        nn_model = baseline_nn(train_X.shape[1])
        nn_model.summary()
        
        from keras.callbacks import EarlyStopping
        from keras.callbacks import ModelCheckpoint  
        
        early_stop_loss = EarlyStopping(monitor='val_loss', patience=15, verbose=0)
        early_stop_acc = EarlyStopping(monitor='val_acc', patience=15, verbose=0)
        
        epochs = 200
        batch_size = 256

        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5', 
                               verbose=1, save_best_only=True)

        hist = nn_model.fit(train_X.loc[:, features].values, train_y.values, validation_split=0.1, 
                            batch_size=batch_size, callbacks=[checkpointer], verbose=1)
        
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(build_fn=baseline_nn, **sk_params)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=5)
        results = np.sqrt(-cross_val_score(pipeline, train_X.loc[:, features].values, train_y.values, cv=kfold, scoring="mean_squared_error"))
        print(results)
        print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
        
        
        