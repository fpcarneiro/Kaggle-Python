import gc
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def display_importances(feature_importance_df_, how_many = 40):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:how_many].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(train_X, train_Y, test_X, num_folds, stratified = False):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_X.shape, test_X.shape))
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_X.shape[0])
    sub_preds = np.zeros(test_X.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_X.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_X[feats], train_Y)):
        train_x, train_y = train_X[feats].iloc[train_idx], train_Y[train_idx]
        valid_x, valid_y = train_X[feats].iloc[valid_idx], train_Y[valid_idx]
        
        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=2,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_X[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    auc_score = roc_auc_score(train_Y, oof_preds)
    print('Full AUC score %.6f' % auc_score)
    # Write submission file and plot feature importance
    #if not debug:
        #file_name = "%s_%.6f.csv" % (prefix_file_name, auc_score)
        #test_X['TARGET'] = sub_preds
        #test_X[['SK_ID_CURR', 'TARGET']].to_csv(file_name, index= False)
    return auc_score, sub_preds, feature_importance_df

def kfold_xgb(train_X, train_Y, test_X, num_folds, stratified = False):
    print("Starting XGB. Train shape: {}, test shape: {}".format(train_X.shape, test_X.shape))
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_X.shape[0])
    sub_preds = np.zeros(test_X.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_X.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_X[feats], train_Y)):
        train_x, train_y = train_X[feats].iloc[train_idx], train_Y[train_idx]
        valid_x, valid_y = train_X[feats].iloc[valid_idx], train_Y[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = XGBClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, ntree_limit=clf.best_iteration)[:, 1]
        sub_preds += clf.predict_proba(test_X[feats], ntree_limit=clf.best_iteration)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    auc_score = roc_auc_score(train_Y, oof_preds)
    print('Full AUC score %.6f' % auc_score)
    # Write submission file and plot feature importance
    #if not debug:
    #    file_name = "%s_%.6f.csv" % (prefix_file_name, auc_score)
    #    test_X['TARGET'] = sub_preds
    #    test_X[['SK_ID_CURR', 'TARGET']].to_csv(file_name, index= False)
    
    return auc_score, sub_preds, feature_importance_df