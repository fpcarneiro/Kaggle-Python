# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:25:09 2019

@author: Y435
"""

import load as ld
import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
from training import save_importances, AveragingModels, LightGBMWrapper, XgbWrapper, GenericWrapper, OOFClassifier
import matplotlib.pyplot as plt
import pandas as pd

def get_tree_models():
    lgb_params = {}
    lgb_params['nthread'] = 3
    lgb_params['n_estimators'] = 10000
    lgb_params['learning_rate'] = 0.02
    lgb_params['colsample_bytree'] = 0.9497036
    lgb_params['subsample'] = 0.8715623
    lgb_params['max_depth'] = 8
    lgb_params["reg_alpha"] = 0.041545473
    lgb_params['reg_lambda'] = 0.0735294
    lgb_params['min_split_gain'] = 0.0222415
    lgb_params['min_child_weight'] = 39.3259775
    lgb_params['num_leaves'] = 34
    lgb_params['metric'] = 'auc'
    
    lgb_fit_params = {}
    lgb_fit_params['verbose_eval'] = 100
    lgb_fit_params['early_stopping_rounds'] = 200
    lgb_fit_params['valid_sets'] = {}
    lgb_fit_params['valid_names'] = ["train", "validation"]

    tree_models = []   

    lgbm = LightGBMWrapper(params = lgb_params, name = "lgbm")
    
    tree_models.append((lgbm, lgb_fit_params))
        
    return tree_models

def run_lgb(train_X, train_y, val_X, val_y, test_X, features = None, verbose = 50, early_stopping_rounds = 200):
    
    lgb_params = {
            "objective" : "regression",
            "metric" : "rmse",
            "num_leaves" : 30,
            "min_child_weight" : 50,
            "learning_rate" : 0.05,
            "bagging_fraction" : 0.7,
            "feature_fraction" : 0.7,
            "bagging_frequency" : 5,
            "bagging_seed" : 2018,
            "num_iterations" : 1000
        }
       
    if features == None:
        features = train_X.columns.tolist()
        
    #train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 2018)
    
    lgb_train = lgb.Dataset(data = train_X, label = train_y, feature_name = features)
    lgb_val = lgb.Dataset(data = val_X, label = val_y, feature_name = features)
    
    lgb_booster = lgb.train(params = lgb_params, train_set = lgb_train, valid_sets = [lgb_train, lgb_val], valid_names = ["train", "validation"], 
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
    verbose = 100
    early_stopping_rounds = 100
    
    with pp.timer("Full Model Run"):
        train_X, train_y, test_X, ids = get_datasets(debug_size = debug_size, silent = silent)
        features = train_X.columns.tolist()
        
        pred_test = 0
        kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)
        for dev_index, val_index in kf.split(train_X):
            dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
            dev_y, val_y = train_y[dev_index], train_y[val_index]
    
            lgb_booster, predictions = run_lgb(dev_X, dev_y, val_X, val_y, test_X, features, verbose = verbose,  early_stopping_rounds = early_stopping_rounds)
            pred_test += predictions
            print("")
            
        pred_test /= 5.
        
        fig, ax = plt.subplots(figsize=(12,10))
        lgb.plot_importance(lgb_booster, max_num_features=50, height=0.8, ax=ax)
        ax.grid(False)
        plt.title("LightGBM - Feature Importance", fontsize=15)
        plt.show()
        
        sub_df = pd.DataFrame({"card_id": ids})
        sub_df["target"] = pred_test
        sub_df.to_csv("first_lgb.csv", index=False)
        
        
        