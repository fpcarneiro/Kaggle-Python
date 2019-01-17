import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import preprocessing as pp
from preprocessing import timer

numeric_agg_funcs = ['mean', 'median', 'sum']

def load_train_test(nrows = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False):
    train, test = pp.read_train_test(nrows = nrows)
    train['first_active_month'] =  pd.to_datetime(train['first_active_month'], format='%Y-%m-%d')
    test['first_active_month'] =  pd.to_datetime(test['first_active_month'], format='%Y-%m-%d')
    
    train["year"] = train["first_active_month"].dt.year
    test["year"] = test["first_active_month"].dt.year
    train["month"] = train["first_active_month"].dt.month
    test["month"] = test["first_active_month"].dt.month
    
    train = pp.hot_encode(train, ["feature_1", "feature_2", "feature_3"])
    test = pp.hot_encode(test, ["feature_1", "feature_2", "feature_3"])
    
    return train, test

def get_processed_files(debug_size, silent = True):
    num_rows = debug_size if debug_size != 0 else None
    with timer("Process train and test"):
        train, test = load_train_test(nrows = num_rows, silent = silent, treat_cat_missing = True, 
                                      treat_num_missing = True, remove_duplicated_cols = True)
        if silent == False:
            print("Train df shape:", train.shape)
            print("Test df shape:", test.shape)

        
    return train, test