import gc
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

def historical_transactions(nrows = None):
    group_var = ['card_id']
    
    cat_columns = ['authorized_flag', 'category_3']
    num_columns = ['installments', 'purchase_amount']
    
    historical_transactions = pp.read_dataset_csv(filename = "historical_transactions.csv", nrows = nrows)
    historical_transactions = pp.reduce_mem_usage(historical_transactions)
    
    counts = pp.get_counts_features(historical_transactions, group_var, "HISTORY")
    
    fe = pp.get_engineered_features(historical_transactions, group_var = group_var, df_name = "HISTORY", num_columns = num_columns, dummy_na = True, cat_columns = cat_columns)
    
    return counts.merge(fe, on = group_var[0], how = 'left')

def get_processed_files(debug_size, silent = True):
    num_rows = debug_size if debug_size != 0 else None
    with timer("Process train and test"):
        train, test = load_train_test(nrows = num_rows, silent = silent, treat_cat_missing = True, 
                                      treat_num_missing = True, remove_duplicated_cols = True)
        subset_ids = list(train.card_id) + list(test.card_id) if debug_size != 0 else None
        if silent == False:
            print("Train shape:", train.shape)
            print("Test shape:", test.shape)
    with timer("Process Historic Transactions"):
        historical_transactions_agg = historical_transactions()
        if silent == False:
           print("Historic Transactions shape:", historical_transactions_agg.shape)
        train = train.merge(historical_transactions_agg, on = 'card_id', how = 'left')
        test = test.merge(historical_transactions_agg, on = 'card_id', how = 'left')
        del historical_transactions_agg
        gc.collect()
        
    return train, test