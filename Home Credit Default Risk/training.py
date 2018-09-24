import gc
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

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
    
def save_importances(features, importances, fold = -1, sort = False, drop_importance_zero = False):
    importance_record = pd.DataFrame()
    importance_record["FEATURE"] = features
    importance_record["IMPORTANCE"] = importances
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
            
            fold_importance_df = save_importances(train_x.columns.tolist(), self.models_[n_fold].feature_importances_, fold = n_fold + 1)
            
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