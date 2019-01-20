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

import warnings
warnings.filterwarnings('ignore')

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
    debug_size = 0
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
                
                del model, pred, cv_score, feat_importance
                gc.collect()
                
                print("*" * 80)
        
#        pred_test = 0
#        kf = model_selection.KFold(n_splits=10, random_state=2018, shuffle=True)
#        for dev_index, val_index in kf.split(train_X):
#            dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
#            dev_y, val_y = train_y[dev_index], train_y[val_index]
#    
#            lgb_booster, predictions = run_lgb(dev_X, dev_y, val_X, val_y, test_X, features, verbose = verbose,  early_stopping_rounds = early_stopping_rounds)
#            pred_test += predictions
#            print("")
#            
#        pred_test /= 5.
#        
#        fig, ax = plt.subplots(figsize=(12,10))
#        lgb.plot_importance(lgb_booster, max_num_features=50, height=0.8, ax=ax)
#        ax.grid(False)
#        plt.title("LightGBM - Feature Importance", fontsize=15)
#        plt.show()
#        
#        sub_df = pd.DataFrame({"card_id": ids})
#        sub_df["target"] = pred_test
#        sub_df.to_csv("first_lgb.csv", index=False)
        
        
        