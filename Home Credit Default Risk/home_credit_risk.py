import pandas as pd
import numpy as np
import preprocessing as pp
import evaluation as ev
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

# Suppress warnings from pandas
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

import gc

import feature_selection as fs

import load as ld

from training import kfold_lightgbm, kfold_xgb, timer, display_importances

from sklearn.preprocessing import Imputer


plt.style.use('fivethirtyeight')



train_X = train.fillna(0)
test_X = test.fillna(0)

# PREPARING TO TRAIN
train_y = train_X['TARGET']
train_X = train_X.drop(['SK_ID_CURR', 'TARGET'], axis=1)

ids = test_X['SK_ID_CURR']
test_X = test_X.drop(['SK_ID_CURR'], axis=1)

duplicated = pp.duplicate_columns(train_X, verbose = True, progress = False)
if len(duplicated)>0:
    train_X.drop(list(duplicated.keys()), axis=1, inplace = True)
    test_X.drop(list(duplicated.keys()), axis=1, inplace = True)

features_variance = fs.list_features_low_variance(train_X, train_y, .98)
train_X_reduced = train_X[features_variance]
test_X_reduced = test_X[features_variance]

del train_X, test_X
gc.collect()

pipeline = Pipeline([
                     ('scaler', MinMaxScaler(feature_range = (0, 1))),
                     #('low_variance', VarianceThreshold(0.98 * (1 - 0.98))),
                     ('reduce_dim', SelectFromModel(lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=1500, objective = 'binary', 
                                   learning_rate = 0.05, silent = False,
                                   subsample = 0.8, colsample_bytree = 0.5))),
                     ])

pipeline.fit(train_X_reduced, train_y)

features_select_from_model = list(train_X_reduced.loc[:, pipeline.named_steps['reduce_dim'].get_support()].columns)

train_X_reduced = pipeline.transform(train_X_reduced)
test_X_reduced = pipeline.transform(test_X_reduced)

###############################################################################
#XGBOOST
###############################################################################

xgb_train = xgb.DMatrix(data=train_X_reduced, label=train_y, feature_names = features_select_from_model)
xg_test = xgb.DMatrix(data=test_X_reduced, feature_names = features_select_from_model)

xgb_params = dict()
xgb_params["booster"] = "gbtree"
xgb_params["objective"] = "binary:logistic"
xgb_params["colsample_bytree"] = 0.4385
xgb_params["subsample"] = 0.7379
xgb_params["max_depth"] = 3
xgb_params['reg_alpha'] = 0.1
xgb_params['reg_lambda'] = 0.1
xgb_params["learning_rate"] = 0.09
xgb_params["min_child_weight"] = 2

xgb_results = xgb.cv(dtrain=xgb_train, params=xgb_params, nfold=3,
                    num_boost_round=1500, early_stopping_rounds=50, metrics="auc", as_pandas=True, seed=2018, verbose_eval = 10)
#xgb_results.head()
#print((xgb_results["test-auc-mean"]).tail(1))

xgbooster = xgb.train(params = xgb_params, dtrain = xgb_train, num_boost_round = 750, maximize = True)

#import matplotlib.pyplot as plt

#xgb.plot_tree(xgbooster,num_trees=0)
#plt.rcParams['figure.figsize'] = [1000, 1000]
#plt.show()

#xgb.plot_importance(xgbooster)
#plt.rcParams['figure.figsize'] = [50, 50]
#plt.show()

pred = xgbooster.predict(xg_test)
my_submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': pred})
my_submission.to_csv("xgb_dmatrix.csv", index=False)

# LIGHT GBM

feat_importance_lgb = kfold_lightgbm(train_X_reduced, train_y, test_X_reduced, num_folds= 5, stratified= False, debug= False)

lgb_train = lgb.Dataset(train_X_reduced, label=train_y, feature_name = features_select_from_model)
lgb_train = lgb.Dataset(train_X_reduced, label=train_y.values, feature_name = list(train_X_reduced.columns))

#lgb_test = lgb.Dataset(test_X_reduced)

lgb_params = {}
lgb_params['boosting_type'] = 'gbdt'
lgb_params['objective'] = 'binary'
lgb_params['learning_rate'] = 0.0596
lgb_params['reg_alpha'] = 0.1
lgb_params['reg_lambda'] = 0.1
lgb_params['max_depth'] = 3
lgb_params['subsample'] = 0.7379
lgb_params["colsample_bytree"] = 0.4385
lgb_params['metric'] = 'auc'

# Params to test later: stratified, shuffle, 
lgb_results = lgb.cv(train_set = lgb_train, params = lgb_params, num_boost_round = 1500, nfold = 3,
       metrics='auc', early_stopping_rounds = 50, verbose_eval = 10, seed=2018)

lgb_booster = lgb.train(params = lgb_params, train_set = lgb_train, num_boost_round = 1450)

lgb_predict = lgb_booster.predict(test_X_reduced)
my_submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': lgb_predict})
my_submission.to_csv("lgb_dataset.csv", index=False)
#[1260]  cv_agg's auc: 0.780272 + 0.00120376

lgb_results = lgb.cv(train_set = lgb_train, params = lgb_grid.best_params_, num_boost_round = 1500, nfold = 3,
       metrics='auc', early_stopping_rounds = 50, verbose_eval = 10, seed=2018)

lgb_booster = lgb.train(params = lgb_grid.best_params_, train_set = lgb_train, num_boost_round = 830)









importances_tree = fs.get_feature_importance(lgb.LGBMClassifier(n_estimators=1500, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = 1, random_state = 50), train_X, train_y)
fs.plot_features_importances(importances_tree, show_importance_zero = False)

def go_cv(trainset_X, trainset_y):
    #model_gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=0.05, max_depth=5, subsample = 0.8, random_state=0)
    #model_logc = LogisticRegression(C = 0.0001)
    #model_rf = RandomForestClassifier(n_estimators = 10, n_jobs = 1)
    model_lgb = lgb.LGBMClassifier(n_estimators=1500, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = 1, random_state = 50)
    model_xgb = xgb.XGBClassifier(colsample_bytree=0.35, gamma=0.027, 
                             learning_rate=0.03, max_depth=4, 
                             min_child_weight=1.7817, n_estimators=1500,
                             reg_alpha=0.43, reg_lambda=0.88,
                             subsample=0.5213, silent=1,
                             random_state = 0, n_jobs = 1)

    models = []
    #models.append(("lr", model_logc))
    #models.append(("gb", model_gbc))
    models.append(("lgb", model_lgb))
    #models.append(("rf", model_rf))
    models.append(("xgb", model_xgb))

    seed = 2018
    results = ev.get_cross_validate(models, trainset_X, trainset_y, 
                                       folds = 3, repetitions = 1, seed = seed, train_score = False)
    return results































debug_size = 0

def process_files(debug_size = 0, treat_duplicated = False, select_features_model = False):
    num_rows = debug_size if debug_size != 0 else None
    with timer("Process application_train and application_test"):
        train, test = ld.load_train_test(nrows = num_rows, silent = True)
        subset_ids = list(train.SK_ID_CURR) + list(test.SK_ID_CURR) if debug_size != 0 else None
        print("Train df shape:", train.shape)
        print("Test df shape:", test.shape)
    with timer("Process Bureau"):
        bureau_agg = ld.bureau(subset_ids, silent = False)
        print("Bureau df shape:", bureau_agg.shape)
        train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
        test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
        del bureau_agg
        gc.collect()
    with timer("Process Bureau Balance"):
        bureau_balance_agg = ld.bureau_balance(subset_ids, silent = False)
        print("Bureau Balance df shape:", bureau_balance_agg.shape)
        train = train.merge(bureau_balance_agg, on = 'SK_ID_CURR', how = 'left')
        test = test.merge(bureau_balance_agg, on = 'SK_ID_CURR', how = 'left')
        del bureau_balance_agg
        gc.collect()
    with timer("Process previous_applications"):
        previous_application_agg = ld.previous_application(subset_ids, silent = False)
        print("Previous applications df shape:", previous_application_agg.shape)
        train = train.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')
        test = test.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')
        del previous_application_agg
        gc.collect()
    with timer("Process POS-CASH balance"):
        cash_balance_agg = ld.cash_balance(subset_ids, silent = False)
        print("Cash Balance df shape:", cash_balance_agg.shape)
        train = train.merge(cash_balance_agg, on = 'SK_ID_CURR', how = 'left')
        test = test.merge(cash_balance_agg, on = 'SK_ID_CURR', how = 'left')
        del cash_balance_agg
        gc.collect()
    with timer("Process credit card balance"):
        credit_balance_agg = ld.credit_balance(subset_ids, silent = False)
        print("Credit Card Balance df shape:", credit_balance_agg.shape)
        train = train.merge(credit_balance_agg, on = 'SK_ID_CURR', how = 'left')
        test = test.merge(credit_balance_agg, on = 'SK_ID_CURR', how = 'left')
        del credit_balance_agg
        gc.collect()
    with timer("Process installments payments"):
        installments_payments_agg = ld.installments_payments(subset_ids, silent = False)
        print("Installments Payments df shape:", installments_payments_agg.shape)
        train = train.merge(installments_payments_agg, on = 'SK_ID_CURR', how = 'left')
        test = test.merge(installments_payments_agg, on = 'SK_ID_CURR', how = 'left')
        del installments_payments_agg
        gc.collect()
    
    train = pp.convert_types(train, print_info = True)
    test = pp.convert_types(test, print_info = True)
    
    train_y = train['TARGET']
    train_X = train.drop(['SK_ID_CURR', 'TARGET'], axis=1)

    ids = test['SK_ID_CURR']
    test_X = test.drop(['SK_ID_CURR'], axis=1)
    if select_features_model:
        with timer("Selecting features with model..."): 
            original_num_columns = len(train_X.columns)
            model = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=1500, objective = 'binary', learning_rate = 0.05, silent = False, subsample = 0.8, colsample_bytree = 0.5)
            pipeline = Pipeline([
                     ('imputer', Imputer(strategy='mean', axis=0, verbose=0)),
                     ('scaler', MinMaxScaler(feature_range = (0, 1))),
                     ('reduce_dim', SelectFromModel(model, threshold = "median")),
                     ])
    
            pipeline.fit(train_X, train_y)

            features_select_from_model = list(train_X.loc[:, pipeline.named_steps['reduce_dim'].get_support()].columns)
            print("Models will be trained with {} out of {} features.".format(len(features_select_from_model), original_num_columns))

            #train = pipeline.transform(train_X)
            #test = pipeline.transform(test_X)
            train_X = train_X.loc[:, list(features_select_from_model)]
            test_X = test_X.loc[:, list(features_select_from_model)]
            
    if treat_duplicated:
        with timer("Treating duplicated"):
            duplicated = pp.duplicate_columns(train, verbose = True, progress = False)
            if len(duplicated) > 0:
                train.drop(list(duplicated.keys()), axis=1, inplace = True)
                test.drop(list(duplicated.keys()), axis=1, inplace = True)
    return train_X, test_X, train_y, ids

if __name__ == "__main__":
    with timer("Full model run"):
        train_X, test_X, train_y, ids = process_files(debug_size = debug_size, treat_duplicated = False, select_features_model = True)
        
        features_variance = fs.list_features_low_variance(train_X.fillna(0), train_y, .95)
        train_X = train_X[features_variance]
        test_X = test_X[features_variance]
        with timer("Run LightGBM with kfold"):
            lgb_cv_score, lgb_pred, feat_importance_lgb = kfold_lightgbm(train_X, train_y, test_X, num_folds= 5, stratified= False)
            display_importances(feat_importance_lgb)
            submit_lgb = pp.submit_file(ids, lgb_pred, prefix_file_name = "lightgbm", cv_score = lgb_cv_score)
        with timer("Run XGB with kfold"):
            xgb_cv_score, xgb_pred, feat_importance_xgb = kfold_xgb(train_X, train_y, test_X, num_folds= 5, stratified= False)
            display_importances(feat_importance_xgb)
            submit_xgb = pp.submit_file(ids, lgb_pred, prefix_file_name = "xgb", cv_score = xgb_cv_score)
    del test_X, train_X, train_y
    del features_variance
    gc.collect()