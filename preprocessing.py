import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

DATADIR = "input/"

def list_files(input_dir = "input/"):
    return (os.listdir(input_dir))

def read_train_test(input_dir = "input/", train_file = 'train.csv', test_file = 'test.csv'):
    train = pd.read_csv(input_dir + train_file)
    test = pd.read_csv(input_dir + test_file)
    return train, test

def get_memory_usage_mb(dataset):
    return (dataset.memory_usage().sum() / 1024 / 1024)

def check_missing(dataset):
    all_data_na_absolute = dataset.isnull().sum()
    all_data_na_percent = (dataset.isnull().sum() / len(dataset)) * 100
    mis_val_table = pd.concat([all_data_na_absolute, all_data_na_percent], axis=1)
    mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'}, inplace = True)
    missing_table = mis_val_table.drop(mis_val_table[mis_val_table.iloc[:, 1] == 0].index).sort_values('% of Total Values', ascending=False).round(2)
    return(missing_table)
    
def get_feature_groups(dataset):
    num_columns = list(dataset.select_dtypes(exclude=['object', 'category']).columns)
    cat_columns = list(dataset.select_dtypes(include=['object', 'category']).columns)
    return (num_columns, cat_columns)

def get_dtypes_columns(dataset):
    groups = dataset.columns.to_series().groupby(dataset.dtypes).groups
    return (groups)

def get_dtype_columns(dataset, dtypes = None):
    types_cols = get_dtypes_columns(dataset)
    if dtypes == None:
        list_columns = [list(cols) for tipo, cols in types_cols.items()]
    else:
        list_columns = [list(cols) for tipo, cols in types_cols.items() if tipo in dtypes]
    columns = []
    for cols in list_columns:
        columns += cols
    return(columns)

def get_missing_cols(dataset, dtypes = None):
    cols = get_dtype_columns(dataset, dtypes = dtypes)
    cols_null = [col for col in cols if dataset[col].isnull().any()]
    return (cols_null)

def get_categorical_missing_cols(dataset):
    cat_cols_null = get_missing_cols(dataset, dtypes = [np.dtype(object)])
    return (cat_cols_null)

def get_numerical_missing_cols(dataset):
    num_cols_null = get_missing_cols(dataset, dtypes = [np.dtype(np.int64), np.dtype(np.float64)])
    return (num_cols_null)

def add_columns_was_missing(X):
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    new_columns = []
    for col in cols_with_missing:
        new_col = col + '_was_missing'
        new_columns.append(new_col)
        X[new_col] = X[col].isnull()
    return X

def handle_missing_mode(X, mode_cols, group_by_cols = None):
        
    if group_by_cols == None:
        #X_ = X.copy()[mode_catcols]
        X_ = X.copy()
        for col in mode_cols:
            if X_[col].isnull().any():
                X_[col] = X_[col].transform(lambda x: x.fillna(x.mode()[0]))
    else:
        #X_ = X.copy()[mode_catcols + group_by_cols]
        X_ = X.copy()
        for col in mode_cols:
            if X_[col].isnull().any():
                X_[col] = X_.groupby(group_by_cols)[col].transform(lambda x: x.fillna(x.mode()[0]))
        
    return(X_)

def handle_missing_median(X, median_cols, group_by_cols = None):
        
    if group_by_cols == None:
        #X_ = X.copy()[mode_catcols]
        X_ = X.copy()
        for col in median_cols:
            if X_[col].isnull().any():
                X_[col] = X_[col].transform(lambda x: x.fillna(x.median()))
    else:
        #X_ = X.copy()[mode_catcols + group_by_cols]
        X_ = X.copy()
        for col in median_cols:
            if X_[col].isnull().any():
                X_[col] = X_.groupby(group_by_cols)[col].transform(lambda x: x.fillna(x.median()))
        
    return(X_)

class HandleMissingModeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return handle_missing_mode(X, mode_catcols = self.cols)

    def fit(self, X, y=None):
        self.cols = get_categorical_missing_cols(X)
        return self
    
class HandleMissingMedianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return handle_missing_median(X, median_cols = self.cols)

    def fit(self, X, y=None):
        self.cols = get_numerical_missing_cols(X)
        return self

def hot_encode(X, columns = None):
    if columns == None:
        return pd.get_dummies(X)
    else:
        return pd.get_dummies(X, columns = columns)
    
def correlation_target(dataset, target = "TARGET"):
    corr = dataset.corr()[target].sort_values(ascending = False)
    return(corr)
    
def correlation_matrix(dataset, target = 'TARGET', nvar = 10):
    corrmat = dataset.corr()
    cols = corrmat.nlargest(nvar + 1, target)[target].index
    cm = np.corrcoef(dataset[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                     annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return list(cols[1:])

def get_domain_knowledge_features(X):
    X_domain = X.copy()
    X_domain['CREDIT_INCOME_PERCENT'] = X_domain['AMT_CREDIT'] / X_domain['AMT_INCOME_TOTAL']
    X_domain['ANNUITY_INCOME_PERCENT'] = X_domain['AMT_ANNUITY'] / X_domain['AMT_INCOME_TOTAL']
    X_domain['CREDIT_TERM'] = X_domain['AMT_ANNUITY'] / X_domain['AMT_CREDIT']
    X_domain['DAYS_EMPLOYED_PERCENT'] = X_domain['DAYS_EMPLOYED'] / X_domain['DAYS_BIRTH']
    return (X_domain)