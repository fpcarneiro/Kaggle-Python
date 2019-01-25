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
    
    test["first_active_month"].fillna(test["first_active_month"].mode().iloc[0], inplace=True)
    
    train['first_active_month'] =  pd.to_datetime(train['first_active_month'], format='%Y-%m-%d')
    test['first_active_month'] =  pd.to_datetime(test['first_active_month'], format='%Y-%m-%d')
    
    train['elapsed_time'] = (datetime.datetime.today() - train['first_active_month']).dt.days
    test['elapsed_time'] = (datetime.datetime.today() - test['first_active_month']).dt.days
    
    train["year"] = train["first_active_month"].dt.year
    test["year"] = test["first_active_month"].dt.year
    train["month"] = train["first_active_month"].dt.month
    test["month"] = test["first_active_month"].dt.month
    train['dayofweek'] = train['first_active_month'].dt.dayofweek
    test['dayofweek'] = test['first_active_month'].dt.dayofweek
    train['weekofyear'] = train['first_active_month'].dt.weekofyear
    test['weekofyear'] = test['first_active_month'].dt.weekofyear
    
    train['outliers'] = 0
    train.loc[train['target'] < -30, 'outliers'] = 1
    
    for f in ['feature_1','feature_2','feature_3']:
        order_label = train.groupby([f])['outliers'].mean()
        train[f + "_"] = train[f].map(order_label)
        test[f + "_"] = test[f].map(order_label)
        
    train.drop(['outliers'], axis = 1, inplace=True)
    
    train = pp.hot_encode(train, ["feature_1", "feature_2", "feature_3"])
    test = pp.hot_encode(test, ["feature_1", "feature_2", "feature_3"])
    
    return train, test

def transactions_read(file = "historical_transactions.csv", nrows = None, card_ids = None):    
    transactions = pp.read_dataset_csv(filename = file, nrows = nrows)
    transactions['purchase_date'] =  pd.to_datetime(transactions['purchase_date'], format='%Y-%m-%d')
    transactions = pp.reduce_mem_usage(transactions)
    transactions.sort_values(by=['card_id', 'purchase_date'], inplace = True)
    
    if card_ids != None:
        print("Selecting subset for debug")
        transactions = transactions.loc[transactions['card_id'].isin(card_ids)]
    
    return transactions

def get_interval_first_total(df, group_var = ['card_id'], date_column = "purchase_date", measure = [("M", "months"), ("D", "days")]):
    first_last_purchase = pp.agg_numeric(df, group_var = group_var, df_name = "FIRST_LAST", num_columns = [date_column], agg_funcs =["first", "last"])
    
    first_last_purchase = df.merge(first_last_purchase, how="left", on=group_var[0])
    
    first_last_purchase.rename(index=str, columns={"FIRST_LAST_" + date_column + "_FIRST": "first_" + date_column}, inplace=True)
    first_last_purchase.rename(index=str, columns={"FIRST_LAST_" + date_column + "_LAST": "last_" + date_column}, inplace=True)
    
    first_last_purchase["interval_since_first"] = first_last_purchase[date_column] - first_last_purchase["first_" + date_column]
    first_last_purchase["interval_total"] = first_last_purchase["last_" + date_column] - first_last_purchase["first_" + date_column]
    
    for m, prefix in measure:
    #historical_transactions["interval_since_first"] = historical_transactions.groupby(['card_id']).purchase_date.apply(lambda x: x - x.iloc[0])
        first_last_purchase[prefix + '_since_first'] = first_last_purchase["interval_since_first"] / np.timedelta64(1, m)
        first_last_purchase[prefix + '_interval_total'] = first_last_purchase["interval_total"] / np.timedelta64(1, m)
    
    first_last_purchase.drop(['interval_since_first', 'interval_total', date_column], axis=1, inplace=True)
    
    return first_last_purchase

def interval_previuos(df, group_var = ['card_id'], date_column = "purchase_date", measure = [("M", "months"), ("W", "weeks"), ("D", "days"), ("h", "hours")]):
    df["previous_purchase"] = df.groupby(["card_id"])[date_column].shift(1)
    df["interval_since_previous"] = df[date_column] - df["previous_purchase"]
    
    for m, prefix in measure:
        df[prefix + '_since_previous'] = df["interval_since_previous"] / np.timedelta64(1, m)
        
    df.drop(["previous_purchase", 'interval_since_previous'], axis=1, inplace=True)

def historical_transactions(nrows = None, card_ids = None):
    historical_transactions = transactions_read(file="historical_transactions.csv", nrows=nrows, card_ids=card_ids)
    return get_transactions_stats(historical_transactions, "HT")

def new_transactions(nrows = None, card_ids = None):
    new_transactions = transactions_read("new_merchant_transactions.csv", nrows=nrows, card_ids=card_ids)
    return get_transactions_stats(new_transactions, "NT")

def get_transactions_stats(df, df_name = "HT"):
    group_var = ['card_id']
    
    cat_columns = ['authorized_flag', 'category_3', 'category_1', 'category_2']
    
    num_columns = ['installments', 'purchase_amount','month_lag', 
                   'days_since_first', 'months_since_first',
                   'hours_since_previous', 'days_since_previous', 'weeks_since_previous', 'months_since_previous', 
                   "month_diff"]
    
    unique_columns = ['merchant_id', 'merchant_category_id', 'state_id', 'city_id', 'subsector_id', 
                      'purchase_month', 'purchase_hour', 'purchase_weekofyear', 'purchase_dayofweek', 'purchase_year', 'purchase_weekend']
    
    #aggs['purchase_amount'] = ['sum','max','min','mean','var']
    #aggs['installments'] = ['sum','max','min','mean','var']
    #aggs['month_lag'] = ['max','min','mean','var']
    #aggs['month_diff'] = ['mean']
    #aggs['weekend'] = ['sum', 'mean']
    
    num_agg_funcs = [np.ptp, 'sum', 'mean', 'max', 'min', 'std']
    
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0})
    
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_year"] = df["purchase_date"].dt.year
    df['purchase_weekofyear'] = df['purchase_date'].dt.weekofyear
    df['purchase_dayofweek'] = df['purchase_date'].dt.dayofweek
    df['purchase_weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['purchase_hour'] = df['purchase_date'].dt.hour
    
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
    
    with pp.timer("Calculating Interval since first"):
        interval_first_total = get_interval_first_total(df.loc[:, [group_var[0], "purchase_date"]])
    with pp.timer("Concatenating"):
        df = pd.concat([df.set_index(group_var[0]), interval_first_total.set_index(group_var[0])], axis = 1).reset_index()
    
    dates_stats = pp.agg_numeric(df, group_var = group_var, df_name = df_name, num_columns = ["first_purchase_date", "last_purchase_date", "days_interval_total"], agg_funcs =["first"])
    df.drop(['first_purchase_date', 'last_purchase_date', "days_interval_total"], axis=1, inplace=True)
    
    with pp.timer("Calculating Interval since last"):
        interval_previuos(df)
        
    dates_stats_cat = pp.agg_categorical(df.loc[:, [group_var[0]] + ["purchase_month", "purchase_year"]], group_var = group_var, df_name = df_name, dummy_na = False, cat_columns = ["purchase_month", "purchase_year"])
        
    df.drop(["purchase_date"], axis=1, inplace=True)
   
    with pp.timer("Calculating Counts"):
        stats = pp.get_counts_features(df.loc[:, [group_var[0]]], group_var = group_var, df_name = df_name, new_column_name = "transactions_count")   
    with pp.timer("Calculating NUNIQUE"):
        #nunique = pp.agg_categorical(df.loc[:, [group_var[0]] + unique_columns], group_var = group_var, df_name = df_name, cat_columns = unique_columns, agg_funcs = ['nunique'], cols_alias = ['NUNIQUE'], to_dummy = False)
        nunique = pp.agg_numeric(df.loc[:, [group_var[0]] + unique_columns], group_var = group_var, df_name = df_name, agg_funcs = ['nunique'], num_columns = unique_columns)
    with pp.timer("Calculating Engineered Features - Numerical"):
        num_agg = pp.agg_numeric(df.loc[:, [group_var[0]] + num_columns], group_var = group_var, df_name = df_name, agg_funcs = num_agg_funcs, num_columns = num_columns)
    with pp.timer("Calculating Engineered Features - Categorical"): 
        cat_agg = pp.agg_categorical(df.loc[:, [group_var[0]] + cat_columns], group_var = group_var, df_name = df_name, dummy_na = False, cat_columns = cat_columns)
    
    stats = pd.concat([ds.set_index(group_var[0]) for ds in [stats, nunique, num_agg, cat_agg, dates_stats, dates_stats_cat]], axis = 1).reset_index()
    stats[df_name + "_purchases_per_day"] = stats[df_name + '_transactions_count']/stats[df_name + "_days_interval_total_FIRST"]
    stats[df_name + "_transactions_per_merchant_cat"] = stats[df_name + '_transactions_count']/stats[df_name + "_merchant_category_id_NUNIQUE"]
    
    del df
    gc.collect()

    return stats

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
        historical_transactions_agg = historical_transactions(card_ids=subset_ids)
        if silent == False:
           print("Historic Transactions shape:", historical_transactions_agg.shape)
        train = train.merge(historical_transactions_agg, on = 'card_id', how = 'left')
        test = test.merge(historical_transactions_agg, on = 'card_id', how = 'left')
        
        train["HT_first_purchase_delay"] = (train["HT_first_purchase_date_FIRST"] - train["first_active_month"]) / np.timedelta64(1, "D")
        test["HT_first_purchase_delay"] = (test["HT_first_purchase_date_FIRST"] - test["first_active_month"]) / np.timedelta64(1, "D")
        
        train['HT_days_no_purchase'] = (datetime.datetime.today() - train['HT_last_purchase_date_FIRST']) / np.timedelta64(1, "D")
        test['HT_days_no_purchase'] = (datetime.datetime.today() - test['HT_last_purchase_date_FIRST']) / np.timedelta64(1, "D")
        
        train['HT_weeks_no_purchase'] = (datetime.datetime.today() - train['HT_last_purchase_date_FIRST']) / np.timedelta64(1, "W")
        test['HT_weeks_no_purchase'] = (datetime.datetime.today() - test['HT_last_purchase_date_FIRST']) / np.timedelta64(1, "W")
        
        train.drop(["HT_first_purchase_date_FIRST", "HT_last_purchase_date_FIRST"], axis = 1, inplace=True)
        test.drop(["HT_first_purchase_date_FIRST", "HT_last_purchase_date_FIRST"], axis = 1, inplace=True)
        
        del historical_transactions_agg
        gc.collect()
    
    with timer("Process New Transactions"):
        new_transactions_agg = new_transactions(card_ids=subset_ids)
        if silent == False:
           print("Historic Transactions shape:", new_transactions_agg.shape)
        train = train.merge(new_transactions_agg, on = 'card_id', how = 'left')
        test = test.merge(new_transactions_agg, on = 'card_id', how = 'left')
        
        train["NT_first_purchase_delay"] = (train["NT_first_purchase_date_FIRST"] - train["first_active_month"]) / np.timedelta64(1, "D")
        test["NT_first_purchase_delay"] = (test["NT_first_purchase_date_FIRST"] - test["first_active_month"]) / np.timedelta64(1, "D")
        
        train['NT_days_no_purchase'] = (datetime.datetime.today() - train['NT_last_purchase_date_FIRST']) / np.timedelta64(1, "D")
        test['NT_days_no_purchase'] = (datetime.datetime.today() - test['NT_last_purchase_date_FIRST']) / np.timedelta64(1, "D")
        
        train['NT_weeks_no_purchase'] = (datetime.datetime.today() - train['NT_last_purchase_date_FIRST']) / np.timedelta64(1, "W")
        test['NT_weeks_no_purchase'] = (datetime.datetime.today() - test['NT_last_purchase_date_FIRST']) / np.timedelta64(1, "W")
        
        train.drop(["NT_first_purchase_date_FIRST", "NT_last_purchase_date_FIRST"], axis = 1, inplace=True)
        test.drop(["NT_first_purchase_date_FIRST", "NT_last_purchase_date_FIRST"], axis = 1, inplace=True)
        
        del new_transactions_agg
        gc.collect()
        
        train['card_id_total'] = train['HT_transactions_count'] + train['NT_transactions_count']
        train['purchase_amount_total'] = train['HT_purchase_amount_SUM'] + train['NT_purchase_amount_SUM']
        
    return train, test