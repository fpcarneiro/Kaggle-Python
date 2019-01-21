import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import preprocessing as pp
from preprocessing import timer
import datetime

numeric_agg_funcs = ['mean', 'median', 'sum']

def elapsed_time(df):
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df

def load_train_test(nrows = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False):
    train, test = pp.read_train_test(nrows = nrows)
    train['first_active_month'] =  pd.to_datetime(train['first_active_month'], format='%Y-%m-%d')
    test['first_active_month'] =  pd.to_datetime(test['first_active_month'], format='%Y-%m-%d')
    
    train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
    test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days
    
    train["year"] = train["first_active_month"].dt.year
    test["year"] = test["first_active_month"].dt.year
    train["month"] = train["first_active_month"].dt.month
    test["month"] = test["first_active_month"].dt.month
    
    train = pp.hot_encode(train, ["feature_1", "feature_2", "feature_3"])
    test = pp.hot_encode(test, ["feature_1", "feature_2", "feature_3"])
    
    return train, test

def historical_transactions_read(nrows = None):    
    historical_transactions = pp.read_dataset_csv(filename = "historical_transactions.csv", nrows = nrows)
    historical_transactions['purchase_date'] =  pd.to_datetime(historical_transactions['purchase_date'], format='%Y-%m-%d')
    historical_transactions = pp.reduce_mem_usage(historical_transactions)
    historical_transactions.sort_values(by=['card_id', 'purchase_date'], inplace = True)
    return historical_transactions

def interval_first(df, group_var = ['card_id'], date_column = "purchase_date", measure = [("D", "days")]):
    last_purchase = pp.agg_numeric(df, group_var = group_var, df_name = "LAST", num_columns = [date_column], agg_funcs =["last"])
    
    last_purchase = df.merge(last_purchase, how="left", on=group_var[0])
    
    last_purchase.rename(index=str, columns={"LAST_" + date_column + "_LAST": "last_" + date_column}, inplace=True)
    last_purchase["interval_since_first"] = last_purchase["last_" + date_column] - last_purchase[date_column]
    
    for m, prefix in measure:
    #historical_transactions["interval_since_first"] = historical_transactions.groupby(['card_id']).purchase_date.apply(lambda x: x - x.iloc[0])
        last_purchase[prefix + '_since_first'] = last_purchase["interval_since_first"] / np.timedelta64(1, m)
    
    last_purchase.drop(["last_" + date_column, 'interval_since_first', date_column], axis=1, inplace=True)
    
    return last_purchase

def historical_transactions(nrows = None):
    group_var = ['card_id']
    
    cat_columns = ['authorized_flag', 'category_3', 'category_1', 'category_2']
    num_columns = ['installments', 'purchase_amount','month_lag', 
                   'hours_since_first', 'days_since_first', 'weeks_since_first',
                   'hours_since_last', 'days_since_last', 'weeks_since_last']
    unique_columns = ['merchant_id', 'merchant_category_id', 'state_id', 'city_id', 'subsector_id']
    
    num_agg_funcs = [np.ptp, 'sum', 'mean', 'max', 'min', 'std']
    
    # Read in Historical Transactions
    historical_transactions = historical_transactions_read(nrows)
    
    
    #historical_transactions["purchase_month"] = historical_transactions["purchase_date"].dt.month
    #historical_transactions["purchase_year"] = historical_transactions["purchase_date"].dt.year
    #historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)//30
    #historical_transactions['month_diff'] += historical_transactions['month_lag']
    
    with pp.timer("Calculating Interval since first"):
        interval_since_first = interval_first(historical_transactions.loc[:, [group_var[0], "purchase_date"]])
    
#    historical_transactions["interval_total_hours"] = historical_transactions.groupby(['card_id']).hours_since_first.agg(lambda x: x.iloc[-1])
#    historical_transactions["interval_total_days"] = historical_transactions.groupby(['card_id']).days_since_first.agg(lambda x: x.iloc[-1])
#    historical_transactions["interval_total_weeks"] = historical_transactions.groupby(['card_id']).weeks_since_first.agg(lambda x: x.iloc[-1])
#    
#    with pp.timer("Calculating Interval since last"):
#        historical_transactions["interval_since_last"] = historical_transactions.groupby(['card_id']).purchase_date.diff()
#        historical_transactions['hours_since_last'] = historical_transactions["interval_since_last"] / np.timedelta64(1,'h')
#        historical_transactions['days_since_last'] = historical_transactions["interval_since_last"] / np.timedelta64(1,'D')
#        historical_transactions['weeks_since_last'] = historical_transactions["interval_since_last"] / np.timedelta64(1,'W')
#    
#    with pp.timer("Calculating Interval Total"):
#        interval = pp.agg_numeric(historical_transactions, group_var = group_var, df_name = "INTERVAL", num_columns = ["hours_since_first", "days_since_first", "weeks_since_first"], agg_funcs =["last"])
#    
#    with pp.timer("Calculating Counts"):
#        counts = pp.get_counts_features(historical_transactions, group_var, "HISTORY")
#    
#    counts = counts.merge(interval, on = group_var[0], how = 'left')
#    
#    with pp.timer("Calculating NUNIQUE"):
#        nunique = pp.agg_categorical(historical_transactions, group_var = group_var, df_name = "HISTORY", cat_columns = unique_columns, agg_funcs = ['nunique'], cols_alias = ['NUNIQUE'], to_dummy = False)
#    
#    with pp.timer("Calculating Engineered Features"):
#        fe = pp.get_engineered_features(historical_transactions, group_var = group_var, df_name = "HISTORY", num_columns = num_columns, num_agg_funcs = num_agg_funcs, dummy_na = True, cat_columns = cat_columns)
#    
#    return (counts.merge(fe, on = group_var[0], how = 'left')).merge(nunique, on = group_var[0], how = 'left')
    return interval_since_first

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

'''
min_max = C_ID_4e6213e9bc.groupby(['card_id']).purchase_date.agg(['min', 'max'])
C_ID_4e6213e9bc = C_ID_4e6213e9bc.merge(min_max, left_on="card_id", right_index=True, how="left")
C_ID_4e6213e9bc["max"] - C_ID_4e6213e9bc["min"]

C_ID_4e6213e9bc["intervalo_rolling"] = C_ID_4e6213e9bc.groupby(['card_id']).purchase_date.rolling(2).
C_ID_4e6213e9bc['intervalo_total_dias_anterior'] = C_ID_4e6213e9bc["intervalo_total"] / np.timedelta64(1,'D')

'''
