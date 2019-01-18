import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def add_columns_was_missing(X):
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    new_columns = []
    for col in cols_with_missing:
        new_col = col + '_was_missing'
        new_columns.append(new_col)
        X[new_col] = X[col].isnull()
    return X

def handle_missing_mode(X, mode_catcols, group_by_cols = None):
        
    if group_by_cols == None:
        X_ = X.copy()[mode_catcols]
        for col in mode_catcols:
            if X_[col].isnull().any():
                X_[col] = X_[col].transform(lambda x: x.fillna(x.mode()[0]))
    else:
        X_ = X.copy()[mode_catcols + group_by_cols]
        for col in mode_catcols:
            if X_[col].isnull().any():
                X_[col] = X_.groupby(group_by_cols)[col].transform(lambda x: x.fillna(x.mode()[0]))
        
    return(X_)

class HandleMissingModeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return handle_missing_mode(X)

    def fit(self, X, y=None):
        return self
 