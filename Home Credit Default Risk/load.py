import numpy as np
import gc
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import preprocessing as pp

def load_train_test(nrows = None, silent = True, treat_cat_missing = False, treat_num_missing = False):
    train, test = pp.read_train_test(train_file = 'application_train.csv', test_file = 'application_test.csv', nrows = nrows)
    
    train = train[train['CODE_GENDER'] != 'XNA']
    
    if not silent:
        print("Train samples: {}, Test samples: {}".format(len(train), len(test)))
    
    # Decrease number of categories in ORGANIZATION_TYPE
    _, cat_cols_train = pp.get_feature_groups(train)
    _, cat_cols_test = pp.get_feature_groups(test)
    
    if not silent:
        print("Decreading the number of categories...")
    
    for col in cat_cols_train:
        cat_values_table_train = pp.check_categorical_cols_values(train, col = col)
        s_low_values_train = set(cat_values_table_train[cat_values_table_train.loc[:, "% of Total"] < 1].index)
        
        cat_values_table_test = pp.check_categorical_cols_values(test, col = col)
        s_low_values_test = set(cat_values_table_test[cat_values_table_test.loc[:, "% of Total"] < 1].index)
        
        l_union = list(s_low_values_train.union(s_low_values_test))
        
        if len(l_union) >= 2:
            if not silent:
                print("Decreasing the number of categories in {}...".format(col))
                print("The following categories will be grouped: {}".format(l_union))
            train.loc[train[col].isin(l_union), col] = "Other 2"
            test.loc[test[col].isin(l_union), col] = "Other 2"
            
    #train.loc[:, 'HOUR_APPR_PROCESS_START'] = train.loc[:, 'HOUR_APPR_PROCESS_START'].astype('object')
    #test.loc[:, 'HOUR_APPR_PROCESS_START'] = test.loc[:, 'HOUR_APPR_PROCESS_START'].astype('object')
    
    train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243
    train["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
    test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243
    test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
    
    cat_cols = pp.get_dtype_columns(train, [np.dtype(object)])
    cat_cols2encode = [c for c in cat_cols if len(train[c].value_counts(dropna=False)) <= 2]
    
    if not silent:
        print("Label encoding {}".format(cat_cols2encode))
    
    le = LabelEncoder()
    for col in cat_cols2encode:
        le.fit(train[col])
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    
    # CATEGORICAL MISSING
    #print(pp.check_missing(train[pp.get_categorical_missing_cols(train)]))
    #print(pp.check_missing(test[pp.get_categorical_missing_cols(test)]))
    if (treat_cat_missing):
        if not silent:
            print("Treating categoricals missing...")
        train.NAME_TYPE_SUITE.fillna("Unaccompanied", inplace= True)
        test.NAME_TYPE_SUITE.fillna("Unaccompanied", inplace= True)
    
    # High density missing categorical columns - deserves a column when performing get_dummies
    # FONDKAPREMONT_MODE, WALLSMATERIAL_MODE, HOUSETYPE_MODE, EMERGENCYSTATE_MODE, OCCUPATION_TYPE
    
    if not silent:
        print("Creating dummies variables...")
    train = pd.get_dummies(train, dummy_na = True)
    test = pd.get_dummies(test, dummy_na = True)
    
    train_labels = train['TARGET']
    train, test = train.align(test, join = 'inner', axis = 1)
    train['TARGET'] = train_labels
    
    # NUMERICAL MISSING
    #print(pp.check_missing(train[pp.get_numerical_missing_cols(train)]))
    #print(pp.check_missing(test[pp.get_numerical_missing_cols(test)]))
    if (treat_num_missing):
        if not silent:
            print("Treating numericals missing...")
        num_missing_trans = pp.HandleMissingMedianTransformer()
        train = num_missing_trans.fit_transform(train)
        test = num_missing_trans.fit_transform(test)
    
    if not silent:
        print("Converting columns types to reduce datasets size...")
    train = pp.convert_types(train, print_info = not silent)
    test = pp.convert_types(test, print_info = not silent)
    
    # FEATURE ENGINEERING
    if not silent:
        print("Feature engineering...")
    train = pp.get_domain_knowledge_features(train)
    test = pp.get_domain_knowledge_features(test)
    
    duplicated_train = pp.duplicate_columns(train, verbose = not silent, progress = False)
    if not silent:
        print("Removing duplicated columns {}".format(duplicated_train))
    train.drop(list(duplicated_train.keys()), axis=1, inplace = True)
    test.drop(list(duplicated_train.keys()), axis=1, inplace = True)
    
    return train, test

def bureau(nrows = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "B", ids2filter = None):
    bureau = pp.read_dataset_csv(filename = "bureau.csv", nrows = nrows)
    
    if (ids2filter != None):
        bureau = bureau.loc[bureau.SK_ID_CURR.isin(ids2filter)]
    
    if (treat_num_missing):
        if not silent:
            print("Treating numericals missing...")
#       bureau = pp.handle_missing_median(bureau, pp.get_numerical_missing_cols(bureau), group_by_cols = ["SK_ID_CURR"])
#       print(pp.check_missing(bureau[pp.get_numerical_missing_cols(bureau)]))
    
    if not silent:
        print("Bureau samples: {}".format(bureau.shape))
    
    # Decrease number of categories in ORGANIZATION_TYPE
    _, cat_cols = pp.get_feature_groups(bureau)
    
    if not silent:
        print("Decreading the number of categories...")
    
    for col in cat_cols:
        cat_values_table = pp.check_categorical_cols_values(bureau, col = col)
        s_low_values = set(cat_values_table[cat_values_table.loc[:, "% of Total"] < 1].index)
        
        if len(s_low_values) >= 2:
            if not silent:
                print("Decreasing the number of categories in {}...".format(col))
                print("The following categories will be grouped: {}".format(s_low_values))
            bureau.loc[bureau[col].isin(s_low_values), col] = "Other 2"
    
    if (treat_num_missing):
        if not silent:
            print("Treating numericals missing...")
        bureau = pp.handle_missing_median(bureau, pp.get_numerical_missing_cols(bureau), group_by_cols = ["CREDIT_TYPE"])
    
    if not silent:
        print("Aggregating BUREAU by categories of 'SK_ID_CURR' and 'CREDIT_ACTIVE'...")
    numeric_cols = pp.get_dtype_columns(bureau, dtypes = [np.dtype(np.int64), np.dtype(np.float64)])
    bureau_cat_num_agg = pp.agg_categorical_numeric(bureau, df_name = df_name + "1", 
                                                    funcs = ['sum', 'mean'], group_var = ['SK_ID_CURR', 'CREDIT_ACTIVE'], 
                                                    target_numvar = numeric_cols)

    if not silent:
        print("Aggregating BUREAU by only 'SK_ID_CURR'...")    
    counts = pp.get_counts_features(bureau, group_var = 'SK_ID_CURR', count_var = 'SK_ID_BUREAU', df_name = df_name + '2')
    bureau_agg = pp.get_engineered_features(bureau.drop(['SK_ID_BUREAU'], axis=1), group_var = 'SK_ID_CURR', df_name = df_name + '2', num_agg_funcs = ['mean', 'median', 'sum'])
    
    bureau_agg = counts.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
    
    if remove_duplicated_cols:
        duplicated_bureau_agg = pp.duplicate_columns(bureau_agg, verbose = not silent, progress = False)
        if not silent:
            print("Removing duplicated columns {}".format(duplicated_bureau_agg))
        if len(duplicated_bureau_agg) > 0:
            bureau_agg.drop(list(duplicated_bureau_agg.keys()), axis=1, inplace = True)

    if remove_duplicated_cols:
        duplicated_bureau_cat_num_agg = pp.duplicate_columns(bureau_cat_num_agg, verbose = not silent, progress = False)
        if not silent:
            print("Removing duplicated columns {}".format(duplicated_bureau_cat_num_agg))
        if len(duplicated_bureau_cat_num_agg) > 0:
            bureau_cat_num_agg.drop(list(duplicated_bureau_cat_num_agg.keys()), axis=1, inplace = True)
        
    return bureau_agg.merge(bureau_cat_num_agg, on = 'SK_ID_CURR', how = 'left')
    
    #return bureau_cat_num_agg
    
def bureau_balance(nrows = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "BB", ids2filter = None):
    group_vars = ['SK_ID_BUREAU', 'SK_ID_CURR']
    
    bureau_balance = pp.read_dataset_csv(filename = "bureau_balance.csv", nrows = nrows)
    bureau = pp.read_dataset_csv(filename = "bureau.csv", nrows = nrows, usecols = group_vars)
    
    if (ids2filter != None):
        bureau = bureau.loc[bureau.SK_ID_CURR.isin(ids2filter)]
        #bureau_balance = bureau_balance.loc[bureau_balance.SK_ID_CURR.isin(ids2filter)]
    
#    bureau_balance = pp.convert_types(bureau_balance, print_info = True)
#    #bureau_balance_agg = pp.aggregate_client(bureau_balance, parent_df = bureau[group_vars], group_vars = group_vars, 
#    #                                         df_names = ['bureau_balance', 'client'])
#   
    df_name_temp = '_'
    bureau_balance_agg = pp.get_engineered_features(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = df_name_temp, num_agg_funcs = ['count', 'min', 'max'], cat_agg_funcs = ['sum'], cols_alias = ['count'])
    cols_status = [c for c in bureau_balance_agg.columns if c.endswith("_count") and c.find("_STATUS_") != -1 and c not in [df_name_temp + "_STATUS_X_count", df_name_temp + "_STATUS_C_count", df_name_temp + "_STATUS_0_count"]]
    # DPD ==Days Past Due
    bureau_balance_agg[df_name_temp + "_DPD_count"] = bureau_balance_agg.loc[:, cols_status].sum(axis=1)
#    #bureau_balance_agg[df_name + "_DPD_PERCENT"] = bureau_balance_agg[df_name + "_DPD_COUNT"]/bureau_balance_agg[df_name + "_MONTHS_BALANCE_count"]
#    #bureau_balance_agg[bureau_balance_agg.SK_ID_BUREAU.isin(bureau.SK_ID_BUREAU) == False]
    bureau_balance_agg = bureau_balance_agg.merge(bureau[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'inner')
    bureau_balance_agg = bureau_balance_agg.drop([group_vars[0]], axis=1)
    bureau_balance_agg_by_client = pp.agg_numeric(bureau_balance_agg, group_var = group_vars[1], df_name = df_name, agg_funcs = ['mean', 'sum', 'median'])
    
    cols_status_percent = [c for c in bureau_balance_agg_by_client.columns if c.endswith("_count_sum") and c.find("_STATUS_") != -1] + [df_name + '_' + df_name_temp + "_DPD_count_sum"]
    for c in cols_status_percent:
        bureau_balance_agg_by_client[c + "_PERCENT"] = bureau_balance_agg_by_client[c]/bureau_balance_agg_by_client[df_name + '_' + df_name_temp + "_MONTHS_BALANCE_count_sum"]
    
    if remove_duplicated_cols:
        duplicated_bureau_balance_agg_by_client = pp.duplicate_columns(bureau_balance_agg_by_client, verbose = not silent, progress = False)
        if len(duplicated_bureau_balance_agg_by_client) > 0:
            bureau_balance_agg_by_client.drop(list(duplicated_bureau_balance_agg_by_client.keys()), axis=1, inplace = True)
            
        #bureau_balance_agg_by_client = pp.convert_types(bureau_balance_agg_by_client, print_info = True)

    return bureau_balance_agg_by_client


def previous_application(nrows = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "PA", ids2filter = None):
    previous_application = pp.read_dataset_csv(filename = "previous_application.csv", nrows = nrows)
    
    if (ids2filter != None):
        previous_application = previous_application.loc[previous_application.SK_ID_CURR.isin(ids2filter)]
    
    if not silent:
        print("Deleting columns with high occurance of nulls...")  
    previous_application.drop(['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'DAYS_FIRST_DRAWING'], axis=1, inplace = True)
    
    #previous_application = pp.handle_missing_median(previous_application, pp.get_numerical_missing_cols(previous_application), group_by_cols = ["SK_ID_CURR"])
    #print(pp.check_missing(previous_application[pp.get_numerical_missing_cols(previous_application)]))
    
    previous_application.loc[:, 'HOUR_APPR_PROCESS_START'] = previous_application.loc[:, 'HOUR_APPR_PROCESS_START'].astype('object')
    
    previous_application.NFLAG_INSURED_ON_APPROVAL.fillna(0, inplace= True)
    previous_application.loc[:, 'NFLAG_INSURED_ON_APPROVAL'] = previous_application.loc[:, 'NFLAG_INSURED_ON_APPROVAL'].astype('int32')
    
    cat_cols2encode = ["NFLAG_INSURED_ON_APPROVAL", "FLAG_LAST_APPL_PER_CONTRACT", "NFLAG_LAST_APPL_IN_DAY"]
    
    le = LabelEncoder()
    for col in cat_cols2encode:
        le.fit(previous_application[col])
        previous_application[col] = le.transform(previous_application[col])
    
    if not silent:
        print("Previous Application samples: {}".format(previous_application.shape))
    
    # Decrease number of categories
    _, cat_cols = pp.get_feature_groups(previous_application)
    
    if not silent:
        print("Decreading the number of categories...")
    
    for col in cat_cols:
        cat_values_table = pp.check_categorical_cols_values(previous_application, col = col)
        s_low_values = set(cat_values_table[cat_values_table.loc[:, "% of Total"] < 1].index)
        
        if len(s_low_values) >= 2:
            if not silent:
                print("Decreasing the number of categories in {}...".format(col))
                print("The following categories will be grouped: {}".format(s_low_values))
            previous_application.loc[previous_application[col].isin(s_low_values), col] = "Other 2"

    previous_application.PRODUCT_COMBINATION.fillna("Other 2", inplace= True)
    
    #previous_application['DAYS_FIRST_DRAWING_ANOM'] = previous_application["DAYS_FIRST_DRAWING"] == 365243
    #previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    
    previous_application['DAYS_FIRST_DUE_ANOM'] = previous_application["DAYS_FIRST_DUE"] == 365243
    previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    
    previous_application['DAYS_LAST_DUE_1ST_VERSION_ANOM'] = previous_application["DAYS_LAST_DUE_1ST_VERSION"] == 365243
    previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    
    previous_application['DAYS_LAST_DUE_ANOM'] = previous_application["DAYS_LAST_DUE"] == 365243
    previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    
    previous_application['DAYS_TERMINATION_ANOM'] = previous_application["DAYS_TERMINATION"] == 365243
    previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
#    
#    #previous_application['APP_CREDIT_PERC'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']

    if (treat_num_missing):
        if not silent:
            print("Treating numericals missing...") 
        previous_application = pp.handle_missing_median(previous_application, pp.get_numerical_missing_cols(previous_application), group_by_cols = ["NAME_CONTRACT_STATUS"])
        print(pp.check_missing(previous_application[pp.get_numerical_missing_cols(previous_application)]))
    
    if not silent:
        print("Aggregating PREVIOUS APPLICATION by categories of 'SK_ID_CURR' and 'NAME_CONTRACT_STATUS'...")
    numeric_cols = pp.get_dtype_columns(previous_application, dtypes = [np.dtype(np.int64), np.dtype(np.float64)])
    previous_application_cat_num_agg = pp.agg_categorical_numeric(previous_application, df_name = df_name + "1", 
                                                    funcs = ['sum', 'mean'], group_var = ['SK_ID_CURR', 'NAME_CONTRACT_STATUS'], 
                                                    target_numvar = numeric_cols)
    if not silent:
        print("Aggregating PREVIOUS APPLICATION by only 'SK_ID_CURR'...")        
    #previous_application_agg = pp.get_engineered_features(previous_application.drop(['SK_ID_PREV'], axis=1), group_var = 'SK_ID_CURR', df_name = 'previous', num_agg_funcs = ['count', 'mean', 'median', 'sum'])
    counts = pp.get_counts_features(previous_application, group_var = 'SK_ID_CURR', count_var = 'SK_ID_PREV', df_name = df_name + '2')
    previous_application_agg = pp.get_engineered_features(previous_application.drop(['SK_ID_PREV'], axis=1), group_var = 'SK_ID_CURR', df_name = df_name + '2', num_agg_funcs = ['mean', 'median', 'sum'])
    
    previous_application_agg = counts.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')
    
    if remove_duplicated_cols:
        duplicated_previous_application_agg = pp.duplicate_columns(previous_application_agg, verbose = not silent, progress = False)
        if not silent:
            print("Removing duplicated columns {}".format(duplicated_previous_application_agg))
        if len(duplicated_previous_application_agg) > 0:
            previous_application_agg.drop(list(duplicated_previous_application_agg.keys()), axis=1, inplace = True)
    
    if remove_duplicated_cols:
        duplicated_previous_application_cat_num_agg = pp.duplicate_columns(previous_application_cat_num_agg, verbose = not silent, progress = False)
        if not silent:
            print("Removing duplicated columns {}".format(duplicated_previous_application_cat_num_agg))
        if len(duplicated_previous_application_cat_num_agg) > 0:
            previous_application_cat_num_agg.drop(list(duplicated_previous_application_cat_num_agg.keys()), axis=1, inplace = True)
#        
#    previous_application_agg_id_columns = list(set([c for c in previous_application_agg.columns if c.startswith("SK_ID_")] + [c for c in previous_application_cat_num_agg.columns if c.startswith("SK_ID_")]))
#    previous_application_agg_columns = list(set([c for c in previous_application_agg.columns if c not in previous_application_agg_id_columns] + [c for c in previous_application_cat_num_agg.columns if c not in previous_application_agg_id_columns]))
#    
#    previous_application_agg = pp.convert_types(previous_application_agg, print_info = True)
#    previous_application_cat_num_agg = pp.convert_types(previous_application_cat_num_agg, print_info = True)
#    
#    train = train.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')
#    test = test.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')
#    
#    train = train.merge(previous_application_cat_num_agg, on = 'SK_ID_CURR', how = 'left')
#    test = test.merge(previous_application_cat_num_agg, on = 'SK_ID_CURR', how = 'left')
#    
#    gc.enable()
#    del previous_application, previous_application_agg
#    del previous_application_nclp_table, s_previous_application, previous_application_npt_table, previous_application_crr_table, previous_application_nts_table, previous_application_ngc_table, 
#    del previous_application_ct_table, previous_application_nsi_table, previous_application_pc_table, cat_cols2encode, numeric_cols
#    del previous_application_cat_num_agg, duplicated_previous_application_agg, duplicated_previous_application_cat_num_agg
#    gc.collect()
    return previous_application_agg.merge(previous_application_cat_num_agg, on = 'SK_ID_CURR', how = 'left')

def pos_cash(nrows = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "CA", ids2filter = None):
    group_vars = ['SK_ID_PREV', 'SK_ID_CURR']
    cash = pp.read_dataset_csv(filename = "POS_CASH_balance.csv", nrows = nrows)
    
    if (ids2filter != None):
        cash = cash.loc[cash.SK_ID_CURR.isin(ids2filter)]
    
    if not silent:
        print("Cash Balance samples: {}".format(cash.shape))
    
    # Decrease number of categories
    _, cat_cols = pp.get_feature_groups(cash)
    
    if not silent:
        print("Decreading the number of categories...")
    
    for col in cat_cols:
        cat_values_table = pp.check_categorical_cols_values(cash, col = col)
        s_low_values = set(cat_values_table[cat_values_table.loc[:, "% of Total"] < 1].index)
        
        if len(s_low_values) >= 2:
            if not silent:
                print("Decreasing the number of categories in {}...".format(col))
                print("The following categories will be grouped: {}".format(s_low_values))
            cash.loc[cash[col].isin(s_low_values), col] = "Other 2"
    
    #cash = pp.convert_types(cash, print_info=True)
    
    if not silent:
        print("Aggregating POS_CASH by categories of 'SK_ID_CURR' and 'NAME_CONTRACT_STATUS'...")
    numeric_cols = pp.get_dtype_columns(cash, dtypes = [np.dtype(np.int64), np.dtype(np.float64)])
    cash_cat_num_agg = pp.agg_categorical_numeric(cash, df_name = df_name + "1", 
                                                    funcs = ['sum', 'mean'], group_var = ['SK_ID_CURR', 'NAME_CONTRACT_STATUS'], 
                                                    target_numvar = numeric_cols)
    
    if not silent:
        print("Aggregating POS_CASH by only 'SK_ID_CURR'...")    
    counts = pp.get_counts_features(cash, group_var = 'SK_ID_CURR', count_var = 'SK_ID_PREV', df_name = df_name + '2')
    cash_agg = pp.get_engineered_features(cash.drop(['SK_ID_PREV'], axis=1), group_var = 'SK_ID_CURR', df_name = df_name + '2', num_agg_funcs = ['mean', 'median', 'sum'])
    
    cash_agg = counts.merge(cash_agg, on = 'SK_ID_CURR', how = 'left')
    
    if remove_duplicated_cols:
        duplicated_cash_agg = pp.duplicate_columns(cash_agg, verbose = not silent, progress = False)
        if not silent:
            print("Removing duplicated columns {}".format(duplicated_cash_agg))
        if len(duplicated_cash_agg) > 0:
            cash_agg.drop(list(duplicated_cash_agg.keys()), axis=1, inplace = True)

    if remove_duplicated_cols:
        duplicated_cash_cat_num_agg = pp.duplicate_columns(cash_cat_num_agg, verbose = not silent, progress = False)
        if not silent:
            print("Removing duplicated columns {}".format(duplicated_cash_cat_num_agg))
        if len(duplicated_cash_cat_num_agg) > 0:
            cash_cat_num_agg.drop(list(duplicated_cash_cat_num_agg.keys()), axis=1, inplace = True)
        
    return cash_agg.merge(cash_cat_num_agg, on = 'SK_ID_CURR', how = 'left')
    
#    df_name_temp = '_'
#    cash_agg = pp.get_engineered_features(cash, group_var = 'SK_ID_PREV', df_name = df_name_temp, num_agg_funcs = ['count', 'min', 'max'], cat_agg_funcs = ['sum'], cols_alias = ['count'])
#    
#    cash_agg = cash_agg.merge(cash[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'inner')
#    cash_agg = cash_agg.drop([group_vars[0]], axis=1)
#    #cash_agg_by_client = pp.agg_numeric(cash_agg, group_var = group_vars[1], df_name = df_name, agg_funcs = ['count', 'mean', 'median', 'sum'])
#
#    cash_agg_by_client = pp.agg_numeric(cash_agg, group_var = group_vars[1], df_name = df_name, agg_funcs = ['mean', 'sum', 'median'])
#    
##    #cash_agg = pp.aggregate_client_2(cash, group_vars = group_vars, df_names = ['cash', 'client'])
##    
##    cash_agg = pp.convert_types(cash_agg, print_info = True)
##    cash_agg_by_client = pp.convert_types(cash_agg_by_client, print_info = True)
##    
##    duplicated_cash_agg_by_client = pp.duplicate_columns(cash_agg_by_client, verbose = True, progress = False)
##    if len(duplicated_cash_agg_by_client) > 0:
##        cash_agg_by_client.drop(list(duplicated_cash_agg_by_client.keys()), axis=1, inplace = True)
##        
##    train = train.merge(cash_agg_by_client, on = 'SK_ID_CURR', how = 'left')
##    test = test.merge(cash_agg_by_client, on = 'SK_ID_CURR', how = 'left')
##    return cash_agg
    