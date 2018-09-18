import gc
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone

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
    
def prepare_train(train_X, test_X, num_folds, stratified = False):
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_X.shape[0])
    sub_preds = np.zeros(test_X.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_X.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    return folds, oof_preds, sub_preds, feature_importance_df, feats

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_gbt(classifier, train_X, train_Y, test_X, num_folds, stratified = False):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_X.shape, test_X.shape))
    
    folds, oof_preds, sub_preds, feature_importance_df, feats = prepare_train(train_X, test_X, num_folds, stratified)
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_X[feats], train_Y)):
        train_x, train_y = train_X[feats].iloc[train_idx], train_Y[train_idx]
        valid_x, valid_y = train_X[feats].iloc[valid_idx], train_Y[valid_idx]
        
        # LightGBM parameters found by Bayesian optimization
        clf = clone(classifier)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)
        
        if(hasattr(clf, "best_iteration_")):
            best_iter = clf.best_iteration_
        elif(hasattr(clf, "best_iteration")):
            best_iter = clf.best_iteration

        #oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=best_iter)[:, 1]
        #sub_preds += clf.predict_proba(test_X[feats], num_iteration=best_iter)[:, 1] / folds.n_splits
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(test_X[feats])[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["name"] = "Name"
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    auc_score = roc_auc_score(train_Y, oof_preds)
    print('Full AUC score %.6f' % auc_score)

    return auc_score, sub_preds, feature_importance_df

def kfold_xgb(train_X, train_Y, test_X, num_folds, stratified = False):
    print("Starting XGB. Train shape: {}, test shape: {}".format(train_X.shape, test_X.shape))
    
    folds, oof_preds, sub_preds, feature_importance_df, feats = prepare_train(train_X, test_X, num_folds, stratified)
    
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
    
    return auc_score, sub_preds, feature_importance_df