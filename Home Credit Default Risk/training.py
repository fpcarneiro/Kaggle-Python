import gc
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def display_importances(feature_importance_df_, how_many = 40):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:how_many].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
    
def get_folds(num_folds, stratified = False):
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    return folds
    
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, model, weights = "same"):
        self.model = model
        self.weights = weights
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y, folds = 5, stratified = False, verbose = 100, early_stopping_rounds = 200):
        self.models_ = [clone(self.model) for f in range(folds)]
        self.importances_df = pd.DataFrame()
        
        folds = get_folds(folds, stratified)
        self.oof_preds = np.zeros(X.shape[0])
        
        self.feats = [f for f in X.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X[self.feats], y)):
            train_x, train_y = X[self.feats].iloc[train_idx], y[train_idx]
            valid_x, valid_y = X[self.feats].iloc[valid_idx], y[valid_idx]
        
            self.models_[n_fold].fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose = verbose, early_stopping_rounds = early_stopping_rounds)
            
            self.oof_preds[valid_idx] = self.models_[n_fold].predict_proba(valid_x)[:, 1]
            
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = self.feats
            fold_importance_df["importance"] = self.models_[n_fold].feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            
            self.importances_df = pd.concat([self.importances_df, fold_importance_df], axis=0)
            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, self.oof_preds[valid_idx])))
            
            del train_x, train_y, valid_x, valid_y
            gc.collect()
        
        self.auc_score = roc_auc_score(y, self.oof_preds)
        print('Full AUC score %.6f' % self.auc_score)
        
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict_proba(self, X):
        self.predictions = np.column_stack([model.predict_proba(X[self.feats])[:, 1] for model in self.models_])
        
        if self.weights == "same":
            return np.mean(self.predictions, axis=1)
        else:
            for i in range(len(self.models_)):
                self.predictions[:, i] *= self.weights[i]
            return np.sum(self.predictions, axis=1)