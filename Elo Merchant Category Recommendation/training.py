import gc
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import lightgbm as lgbm
import xgboost as xgb

warnings.simplefilter(action='ignore', category=FutureWarning)

class GenericWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, clf = LogisticRegression(), name = "classifier"):
        self.clf = clf
        self.name = name
        
    def get_params(self, deep=True):
        return {"clf": self.clf, "name": self.name}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, **fit_params):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.feature_names_ = X.columns.tolist()
        self.clf.fit(X, y, **fit_params)
        return self
        
    def predict_proba(self, X):
        check_is_fitted(self, ['classes_', 'feature_names_'])
        return self.clf.predict_proba(X)[:,1]
    
    def feature_importance(self, importance_type='gain', iteration=-1):
        check_is_fitted(self, ['classes_', 'feature_names_'])
        if(hasattr(self.clf, "feature_importances_")):
            return dict(zip(self.feature_names_, self.clf.feature_importances_))
        else:
            return dict(zip(self.feature_names_, [0] * len(self.feature_names_)))

class LightGBMClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, params = {}, name = "lgb"):
        self.params = params
        self.name = name
        
    def get_params(self, deep=True):
        return {"params": self.params, "name": self.name}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, **fit_params):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.feature_names_ = X.columns.tolist()
        
        train_X = lgbm.Dataset(data = X, label = y, feature_name = self.feature_names_)
        
        if 'valid_sets' in fit_params:
            datasets = [ lgbm.Dataset(data = val_x, label = val_y, feature_name = self.feature_names_) for (val_x, val_y) in fit_params['valid_sets'] ]
            fit_params['valid_sets'] = datasets
            self.fit_params_ = fit_params
        
        self.booster_ = lgbm.train(params=self.params, train_set=train_X, **self.fit_params_)
        return self
        
    def predict_proba(self, X):
        check_is_fitted(self, ['booster_', 'classes_', 'feature_names_'])
        return self.booster_.predict(X)
    
    def feature_importance(self, importance_type='gain', iteration=-1):
        check_is_fitted(self, ['booster_', 'classes_', 'feature_names_'])
        return dict(zip(self.feature_names_, self.booster_.feature_importance()))
    
class LightGBMRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, params = {}, name = "lgb"):
        self.params = params
        self.name = name
        
    def get_params(self, deep=True):
        return {"params": self.params, "name": self.name}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, **fit_params):
        self.feature_names_ = X.columns.tolist()
        
        train_X = lgbm.Dataset(data = X, label = y, feature_name = self.feature_names_)
        
        if 'valid_sets' in fit_params:
            datasets = [ lgbm.Dataset(data = val_x, label = val_y, feature_name = self.feature_names_) for (val_x, val_y) in fit_params['valid_sets'] ]
            fit_params['valid_sets'] = datasets
            self.fit_params_ = fit_params
        
        self.booster_ = lgbm.train(params=self.params, train_set=train_X, **self.fit_params_)
        return self
        
    def predict(self, X):
        check_is_fitted(self, ['booster_', 'feature_names_'])
        return self.booster_.predict(X)
    
    def feature_importance(self, importance_type='gain', iteration=-1):
        check_is_fitted(self, ['booster_', 'feature_names_'])
        return dict(zip(self.feature_names_, self.booster_.feature_importance()))
    
class XgbWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, params = {}, name = "xgb"):
        self.params = params
        self.name = name
        
    def get_params(self, deep=True):
        return {"params": self.params, "name": self.name}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, **fit_params):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.feature_names_ = X.columns.tolist()
        
        train_X = xgb.DMatrix(data=X, label=y, feature_names = self.feature_names_)
        
        if 'evals' in fit_params:
            datasets = [ (xgb.DMatrix(data = val_x, label = val_y, feature_names = self.feature_names_), name) for (val_x, val_y), name in fit_params['evals'] ]
            fit_params['evals'] = datasets
            self.fit_params_ = fit_params
        
        self.booster_ = xgb.train(params=self.params, dtrain=train_X, **fit_params)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['booster_', 'classes_', 'feature_names_'])
        return self.booster_.predict(xgb.DMatrix(data=X))
    
    def feature_importance(self, fmap='', importance_type='gain'):
        check_is_fitted(self, ['booster_', 'classes_', 'feature_names_'])
        return self.booster_.get_score()
    
class XgbRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, params = {}, name = "xgb"):
        self.params = params
        self.name = name
        
    def get_params(self, deep=True):
        return {"params": self.params, "name": self.name}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, **fit_params):
        self.feature_names_ = X.columns.tolist()
        
        train_X = xgb.DMatrix(data=X, label=y, feature_names = self.feature_names_)
        
        if 'evals' in fit_params:
            datasets = [ (xgb.DMatrix(data = val_x, label = val_y, feature_names = self.feature_names_), name) for (val_x, val_y), name in fit_params['evals'] ]
            fit_params['evals'] = datasets
            self.fit_params_ = fit_params
        
        self.booster_ = xgb.train(params=self.params, dtrain=train_X, **fit_params)
        return self

    def predict(self, X):
        check_is_fitted(self, ['booster_', 'feature_names_'])
        return self.booster_.predict(xgb.DMatrix(data=X))
    
    def feature_importance(self, fmap='', importance_type='gain'):
        check_is_fitted(self, ['booster_', 'feature_names_'])
        return self.booster_.get_score()


def display_importances(feature_importance_df_, how_many = 40):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:how_many].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
    
def save_importances(importances, fold = -1, sort = False, drop_importance_zero = False):
    importance_record = pd.DataFrame()
    importance_record["FEATURE"] = importances.keys()
    importance_record["IMPORTANCE"] = importances.values()
    
    if fold != -1:
        importance_record["FOLD"] = fold
    if sort:
        importance_record.sort_values(by = "IMPORTANCE", ascending=False, inplace = True)
    if drop_importance_zero:
        importance_record = importance_record[importance_record.IMPORTANCE != 0]
    
    return importance_record
    
def get_folds(num_folds, stratified = False, seed = 1001):
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    return folds

class OOFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, weights = "same", nfolds = 5, stratified = False):
        self.clf = clf
        self.weights = weights
        self.nfolds = nfolds
        self.stratified = stratified
        
    def get_params(self, deep=True):
        return {"clf": self.clf, "weights": self.weights, "nfolds": self.nfolds, "stratified": self.stratified}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, **fit_params):
        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_, y = np.unique(y, return_inverse=True)
        
        self.models_ = [clone(self.clf) for f in range(self.nfolds)]
        #self.models_ = [None for f in range(self.nfolds)]
        self.importances_ = pd.DataFrame()
        
        folds = get_folds(self.nfolds, self.stratified)
        self.oof_preds_ = np.zeros(X.shape[0])
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            train_x, train_y = X.iloc[train_idx], y[train_idx]
            valid_x, valid_y = X.iloc[valid_idx], y[valid_idx]
            
            if 'valid_sets' in fit_params: # LightGBM
                #fit_params['valid_sets'] = [(train_x, train_y), (valid_x, valid_y)]
                fit_params['valid_sets'] = [(valid_x, valid_y)]
            if 'evals' in fit_params: # XGBoost
                #fit_params['evals'] = [((train_x, train_y), "train"), ((valid_x, valid_y), "validation")]
                fit_params['evals'] = [((valid_x, valid_y), "validation")]
            if 'eval_set' in fit_params:
                fit_params['eval_set'] = [(valid_x, valid_y)]
            
            self.models_[n_fold].fit(train_x, train_y, **fit_params)
            
            self.oof_preds_[valid_idx] = self.models_[n_fold].predict_proba(valid_x)
            
            if(hasattr(self.models_[n_fold], "feature_importances_")):
                importances = self.models_[n_fold].feature_importances_
            elif(hasattr(self.models_[n_fold], "feature_importance")):
                importances = self.models_[n_fold].feature_importance()
            elif(hasattr(self.models_[n_fold], "coef_")):
                importances = self.models_[n_fold].coef_
            elif(hasattr(self.models_[n_fold], "coefs_")):
                importances = self.models_[n_fold].coefs_
            else:
                importances = None
                
            fold_importance_df = save_importances(importances, fold = n_fold + 1)
            
            self.importances_ = pd.concat([self.importances_, fold_importance_df], axis=0)
            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, self.oof_preds_[valid_idx])))
            
            del train_x, train_y, valid_x, valid_y
            gc.collect()
        
        self.auc_score_ = roc_auc_score(y, self.oof_preds_)
        print('Full AUC score %.6f' % self.auc_score_)
        
        return self
    
    def predict_proba(self, X):
        
        # Check is fit had been called
        check_is_fitted(self, ['classes_', 'models_', 'importances_', 'oof_preds_', 'auc_score_'])
        
        self.predictions = np.column_stack([model.predict_proba(X) for model in self.models_])
        
        if self.weights == "same":
            return np.mean(self.predictions, axis=1)
        else:
            for i in range(len(self.models_)):
                self.predictions[:, i] *= self.weights[i]
            return np.sum(self.predictions, axis=1)

class OOFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, reg, weights = "same", nfolds = 5, stratified = False):
        self.reg = reg
        self.weights = weights
        self.nfolds = nfolds
        self.stratified = stratified
        
    def get_params(self, deep=True):
        return {"reg": self.reg, "weights": self.weights, "nfolds": self.nfolds, "stratified": self.stratified}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, **fit_params):
        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)
        # Store the classes seen during fit
        #self.classes_, y = np.unique(y, return_inverse=True)
        
        self.models_ = [clone(self.reg) for f in range(self.nfolds)]
        #self.models_ = [None for f in range(self.nfolds)]
        self.importances_ = pd.DataFrame()
        
        folds = get_folds(self.nfolds, self.stratified)
        self.oof_preds_ = np.zeros(X.shape[0])
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            train_x, train_y = X.iloc[train_idx], y[train_idx]
            valid_x, valid_y = X.iloc[valid_idx], y[valid_idx]
            
            if 'valid_sets' in fit_params: # LightGBM
                #fit_params['valid_sets'] = [(train_x, train_y), (valid_x, valid_y)]
                fit_params['valid_sets'] = [(valid_x, valid_y)]
            if 'evals' in fit_params: # XGBoost
                #fit_params['evals'] = [((train_x, train_y), "train"), ((valid_x, valid_y), "validation")]
                fit_params['evals'] = [((valid_x, valid_y), "validation")]
            if 'eval_set' in fit_params:
                fit_params['eval_set'] = [(valid_x, valid_y)]
            
            self.models_[n_fold].fit(train_x, train_y, **fit_params)
            
            self.oof_preds_[valid_idx] = self.models_[n_fold].predict(valid_x)
            
            if(hasattr(self.models_[n_fold], "feature_importances_")):
                importances = self.models_[n_fold].feature_importances_
            elif(hasattr(self.models_[n_fold], "feature_importance")):
                importances = self.models_[n_fold].feature_importance()
            elif(hasattr(self.models_[n_fold], "coef_")):
                importances = self.models_[n_fold].coef_
            elif(hasattr(self.models_[n_fold], "coefs_")):
                importances = self.models_[n_fold].coefs_
            else:
                importances = None
                
            fold_importance_df = save_importances(importances, fold = n_fold + 1)
            
            self.importances_ = pd.concat([self.importances_, fold_importance_df], axis=0)
            print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(valid_y, self.oof_preds_[valid_idx]))))
            
            del train_x, train_y, valid_x, valid_y
            gc.collect()
        
        self.rmse_score_ = np.sqrt(mean_squared_error(y, self.oof_preds_))
        print('Full RMSE score %.6f' % self.rmse_score_)
        
        return self
    
    def predict(self, X):
        
        # Check is fit had been called
        check_is_fitted(self, ['models_', 'importances_', 'oof_preds_', 'rmse_score_'])
        
        self.predictions = np.column_stack([model.predict(X) for model in self.models_])
        
        if self.weights == "same":
            return np.mean(self.predictions, axis=1)
        else:
            for i in range(len(self.models_)):
                self.predictions[:, i] *= self.weights[i]
            return np.sum(self.predictions, axis=1)
    
class AveragingModels(BaseEstimator, ClassifierMixin):
    def __init__(self, model, weights = "same", nfolds = 5, stratified = False):
        self.model = model
        self.weights = weights
        self.nfolds = nfolds
        self.stratified = stratified
        
    def get_params(self, deep=True):
        return {"model": self.model, "weights": self.weights, "nfolds": self.nfolds, "stratified": self.stratified}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y, **fit_params):
        
        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_, y = np.unique(y, return_inverse=True)
        
        self.models_ = [clone(self.model) for f in range(self.nfolds)]
        self.importances_ = pd.DataFrame()
        
        folds = get_folds(self.nfolds, self.stratified)
        self.oof_preds_ = np.zeros(X.shape[0])
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            train_x, train_y = X.iloc[train_idx], y[train_idx]
            valid_x, valid_y = X.iloc[valid_idx], y[valid_idx]
            
            if 'eval_set' in fit_params:
                fit_params['eval_set'] = [(train_x, train_y), (valid_x, valid_y)]
            
            self.models_[n_fold].fit(train_x, train_y, **fit_params)
            
            self.oof_preds_[valid_idx] = self.models_[n_fold].predict_proba(valid_x)[:, 1]
            
            if(hasattr(self.models_[n_fold], "feature_importances_")):
                importances = self.models_[n_fold].feature_importances_
            elif(hasattr(self.models_[n_fold], "feature_importance")):
                importances = self.models_[n_fold].feature_importance
            elif(hasattr(self.models_[n_fold], "coef_")):
                importances = self.models_[n_fold].coef_
            elif(hasattr(self.models_[n_fold], "coefs_")):
                importances = self.models_[n_fold].coefs_
            else:
                importances = None
                
            fold_importance_df = save_importances(train_x.columns.tolist(), importances, fold = n_fold + 1)
            
            self.importances_ = pd.concat([self.importances_, fold_importance_df], axis=0)
            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, self.oof_preds_[valid_idx])))
            
            del train_x, train_y, valid_x, valid_y
            gc.collect()
        
        self.auc_score_ = roc_auc_score(y, self.oof_preds_)
        print('Full AUC score %.6f' % self.auc_score_)
        
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict_proba(self, X):
        
        # Check is fit had been called
        check_is_fitted(self, ['classes_', 'models_', 'importances_', 'oof_preds_', 'auc_score_'])
        
        self.predictions = np.column_stack([model.predict_proba(X)[:, 1] for model in self.models_])
        
        if self.weights == "same":
            return np.mean(self.predictions, axis=1)
        else:
            for i in range(len(self.models_)):
                self.predictions[:, i] *= self.weights[i]
            return np.sum(self.predictions, axis=1)
        
    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]