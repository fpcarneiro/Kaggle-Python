import numpy as np
import gc
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import preprocessing as pp
from preprocessing import timer

numeric_agg_funcs = ['mean', 'median', 'sum']

def treat_anomalies(df, columns):
    df_copy = df.copy()
    for col in columns:
        df_copy[col + '_ANOM'] = df_copy[col] == 365243
        df_copy[col].replace(365243, np.nan, inplace= True)
    return df_copy

def load_train_test(nrows = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False):
    train, test = pp.read_train_test(train_file = 'application_train.csv', test_file = 'application_test.csv', nrows = nrows)
    
    # Remove some rows with values not present in test set
    train = train[train['CODE_GENDER'] != 'XNA']
    train = train[train['NAME_INCOME_TYPE'] != 'Maternity leave']
    train = train[train['NAME_FAMILY_STATUS'] != 'Unknown']
    
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
            
    train.loc[:, 'HOUR_APPR_PROCESS_START'] = train.loc[:, 'HOUR_APPR_PROCESS_START'].astype('object')
    test.loc[:, 'HOUR_APPR_PROCESS_START'] = test.loc[:, 'HOUR_APPR_PROCESS_START'].astype('object')
       
    train = treat_anomalies(train, columns = ['DAYS_EMPLOYED'])
    test = treat_anomalies(test, columns = ['DAYS_EMPLOYED'])
    
    train.loc[train['OWN_CAR_AGE'] > 80, 'OWN_CAR_AGE'] = np.nan
    train.loc[train['REGION_RATING_CLIENT_W_CITY'] < 0, 'REGION_RATING_CLIENT_W_CITY'] = np.nan
    train.loc[train['AMT_INCOME_TOTAL'] > 1e8, 'AMT_INCOME_TOTAL'] = np.nan
    train.loc[train['AMT_REQ_CREDIT_BUREAU_QRT'] > 10, 'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    train.loc[train['OBS_30_CNT_SOCIAL_CIRCLE'] > 40, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    
    test.loc[test['OWN_CAR_AGE'] > 80, 'OWN_CAR_AGE'] = np.nan
    test.loc[test['REGION_RATING_CLIENT_W_CITY'] < 0, 'REGION_RATING_CLIENT_W_CITY'] = np.nan
    test.loc[test['AMT_INCOME_TOTAL'] > 1e8, 'AMT_INCOME_TOTAL'] = np.nan
    test.loc[test['AMT_REQ_CREDIT_BUREAU_QRT'] > 10, 'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    test.loc[test['OBS_30_CNT_SOCIAL_CIRCLE'] > 40, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    
    train['COUNT_MISSING'] = train.isnull().sum(axis = 1).values
    test['COUNT_MISSING'] = test.isnull().sum(axis = 1).values
    
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
    train = pd.get_dummies(train, dummy_na = treat_cat_missing, dtype = 'bool')
    test = pd.get_dummies(test, dummy_na = treat_cat_missing, dtype = 'bool')
    
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
    
    # FEATURE ENGINEERING
    if not silent:
        print("Feature engineering...")
    train = pp.get_domain_knowledge_features(train)
    test = pp.get_domain_knowledge_features(test)
    
    if remove_duplicated_cols:
        duplicated_train = pp.duplicate_columns(train, verbose = not silent, progress = False)
        if not silent:
            print("Removing duplicated columns {}".format(duplicated_train))
        train.drop(list(duplicated_train.keys()), axis=1, inplace = True)
        test.drop(list(duplicated_train.keys()), axis=1, inplace = True)
    
    return train, test

def bureau(subset_ids = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "BU"):
    group_var = ['SK_ID_CURR']
    
    bureau = pp.read_dataset_csv(filename = "bureau.csv")
    
    if subset_ids != None:
        bureau = bureau.loc[bureau.SK_ID_CURR.isin(subset_ids)]
    
    if not silent:
        print("Bureau Shape: {}".format(bureau.shape))
        
    # Decrease number of categories   
    bureau = pp.join_low_occurance_categories(bureau, silent, join_category_name = "Other 2")
    
    if (treat_num_missing):
        if not silent:
            print("Treating numericals missing...")
        bureau = pp.handle_missing_median(bureau, pp.get_numerical_missing_cols(bureau), group_by_cols = ["CREDIT_TYPE"])
    
    if not silent:
        print("Aggregating BUREAU by categories of 'SK_ID_CURR' and 'CREDIT_ACTIVE'...")
    bu_agg_1 = pp.agg_categorical_numeric(bureau, df_name + "1", 
                                          group_var = ['SK_ID_CURR', 'CREDIT_ACTIVE'], 
                                          num_columns = ['DAYS_CREDIT', 'AMT_ANNUITY'])
    
    if not silent:
        print("Aggregating BUREAU by only 'SK_ID_CURR'...")
    counts = pp.get_counts_features(bureau, group_var, df_name + "2")
    bu_agg_2 = pp.get_engineered_features(bureau, group_var, df_name + "2", num_agg_funcs = numeric_agg_funcs)
    
    bu_agg_2 = counts.merge(bu_agg_2, on = group_var[0], how = 'left')

    return bu_agg_1.merge(bu_agg_2, on = group_var[0], how = 'left')

def previous_application(subset_ids = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "PA"):
    group_var = ['SK_ID_CURR']
    
    previous_application = pp.read_dataset_csv(filename = "previous_application.csv")
    
    if subset_ids != None:
        previous_application = previous_application.loc[previous_application.SK_ID_CURR.isin(subset_ids)]
    
    if not silent:
        print("Previous Application Shape: {}".format(previous_application.shape))
        
    if not silent:
        print("Deleting columns with high occurance of nulls...")  
    previous_application.drop(['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED'], axis=1, inplace = True)
    #previous_application.drop(['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'DAYS_FIRST_DRAWING'], axis=1, inplace = True)
    
    previous_application.NFLAG_INSURED_ON_APPROVAL.fillna(0, inplace= True)
    previous_application.loc[:, 'NFLAG_INSURED_ON_APPROVAL'] = previous_application.loc[:, 'NFLAG_INSURED_ON_APPROVAL'].astype('int32')
    
    previous_application['APP_CREDIT_PERC'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']
    
    # Label Encode
    previous_application = pp.label_encode(previous_application, silent)
    
    # Decrease number of categories   
    previous_application = pp.join_low_occurance_categories(previous_application, silent, join_category_name = "Other 2")
    
    previous_application.PRODUCT_COMBINATION.fillna("Other 2", inplace= True)
        
    previous_application = treat_anomalies(previous_application, columns = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION'])
    
    if (treat_num_missing):
        if not silent:
            print("Treating numericals missing...") 
            print(pp.check_missing(previous_application[pp.get_numerical_missing_cols(previous_application)]))
        
        previous_application = pp.handle_missing_median(previous_application, pp.get_numerical_missing_cols(previous_application), group_by_cols = ["NAME_CONTRACT_STATUS"])
        
        if not silent:
            print(pp.check_missing(previous_application[pp.get_numerical_missing_cols(previous_application)]))
    
    if not silent:
        print("Aggregating PREVIOUS APPLICATION by categories of 'SK_ID_CURR' and 'NAME_CONTRACT_STATUS'...")
    pa_agg_1 = pp.agg_categorical_numeric(previous_application, df_name + "1", 
                                          group_var = ['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
        
    if not silent:
        print("Aggregating PREVIOUS APPLICATION by only 'SK_ID_CURR'...")
    counts = pp.get_counts_features(previous_application, group_var, df_name + "2")
    pa_agg_2 = pp.get_engineered_features(previous_application, group_var, df_name + "2", num_agg_funcs = numeric_agg_funcs)
        
    pa_agg_2 = counts.merge(pa_agg_2, on = group_var[0], how = 'left')
    
    return pa_agg_1.merge(pa_agg_2, on = group_var[0], how = 'left')

def bureau_balance(subset_ids = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "BB"):
    group_vars = ['SK_ID_CURR', 'SK_ID_BUREAU']
    
    bureau_balance = pp.read_dataset_csv(filename = "bureau_balance.csv")
    bureau = pp.read_dataset_csv(filename = "bureau.csv", usecols = group_vars)
    bureau_balance = bureau.merge(bureau_balance, on = 'SK_ID_BUREAU', how = 'inner')
    
    del bureau
    gc.collect()
    
    if subset_ids != None:
        bureau_balance = bureau_balance.loc[bureau_balance.SK_ID_CURR.isin(subset_ids)]
    
    if not silent:
        print("Bureau Balance Shape: {}".format(bureau_balance.shape))
        
    # Decrease number of categories   
    bureau_balance = pp.join_low_occurance_categories(bureau_balance, silent, join_category_name = "Other 2")
    
    df_name_temp = ""
    counts_bb = pp.get_counts_features(bureau_balance, group_vars, df_name_temp, group_vars[1])
    bb_agg = pp.get_engineered_features(bureau_balance, group_vars, df_name_temp, num_agg_funcs = numeric_agg_funcs)
    cols_status = [c for c in bb_agg.columns if c.endswith("_COUNT") and c.find("_STATUS_") != -1 and c not in [df_name_temp + "_STATUS_X_COUNT", df_name_temp + "_STATUS_C_COUNT", df_name_temp + "_STATUS_0_COUNT"]]
    
    bb_agg = counts_bb.merge(bb_agg, on = group_vars, how = 'left')
    
    bb_agg[df_name_temp + "_DPD_COUNT"] = bb_agg.loc[:, cols_status].sum(axis=1)
    bb_agg[df_name_temp + "_DPD_FREQ"] = bb_agg[df_name_temp + "_DPD_COUNT"] / bb_agg[df_name_temp + "_ROWCOUNT"]
       
    bb_agg_client = pp.agg_numeric(bb_agg, [group_vars[0]], df_name)
    
    return bb_agg_client

def cash_balance(subset_ids = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "CB"):
    group_vars = ['SK_ID_CURR', 'SK_ID_PREV']
    
    cash_balance = pp.read_dataset_csv(filename = "POS_CASH_balance.csv")
    
    if subset_ids != None:
        cash_balance = cash_balance.loc[cash_balance.SK_ID_CURR.isin(subset_ids)]
    
    if not silent:
        print("Cash Balance Shape: {}".format(cash_balance.shape))
        
    # Decrease number of categories   
    cash_balance = pp.join_low_occurance_categories(cash_balance, silent, join_category_name = "Other 2")

    df_name_temp = ""
    counts_cb = pp.get_counts_features(cash_balance, group_vars, df_name_temp, group_vars[1])
    cb_agg = pp.get_engineered_features(cash_balance, group_vars, df_name_temp, num_agg_funcs = numeric_agg_funcs)

    cb_agg = counts_cb.merge(cb_agg, on = group_vars, how = 'left')
    
    cb_agg_client = pp.agg_numeric(cb_agg, [group_vars[0]], df_name)
    
    return cb_agg_client

def credit_balance(subset_ids = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "CCB"):
    group_vars = ['SK_ID_CURR', 'SK_ID_PREV']
    
    credit_balance = pp.read_dataset_csv(filename = "credit_card_balance.csv")
    
    if subset_ids != None:
        credit_balance = credit_balance.loc[credit_balance.SK_ID_CURR.isin(subset_ids)]
    
    if not silent:
        print("Credit Card Balance Shape: {}".format(credit_balance.shape))
        
    # Decrease number of categories   
    credit_balance = pp.join_low_occurance_categories(credit_balance, silent, join_category_name = "Other 2")

    df_name_temp = ""
    counts_ccb = pp.get_counts_features(credit_balance, group_vars, df_name_temp, group_vars[1])
    ccb_agg = pp.get_engineered_features(credit_balance, group_vars, df_name_temp, num_agg_funcs = numeric_agg_funcs)

    ccb_agg = counts_ccb.merge(ccb_agg, on = group_vars, how = 'left')
    
    ccb_agg_client = pp.agg_numeric(ccb_agg, [group_vars[0]], df_name)
    
    return ccb_agg_client

def installments_payments(subset_ids = None, silent = True, treat_cat_missing = False, treat_num_missing = False, remove_duplicated_cols = False, df_name = "IP"):
    group_vars = ['SK_ID_CURR', 'SK_ID_PREV']
    
    installments = pp.read_dataset_csv(filename = "installments_payments.csv")
    
    if subset_ids != None:
        installments = installments.loc[installments.SK_ID_CURR.isin(subset_ids)]
    
    if not silent:
        print("Installment Payments Shape: {}".format(installments.shape))
    
    # Percentage and difference paid in each installment (amount paid and installment value)
    installments['PAYMENT_PERC'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
    installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    
    # Days past due and days before due (no negative values)
    installments['DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['DBD'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']
    installments['DPD'] = installments['DPD'].apply(lambda x: x if x > 0 else 0)
    installments['DBD'] = installments['DBD'].apply(lambda x: x if x > 0 else 0)
        
    # Decrease number of categories   
    installments = pp.join_low_occurance_categories(installments, silent, join_category_name = "Other 2")

    df_name_temp = ""
    counts_ip = pp.get_counts_features(installments, group_vars, df_name_temp, group_vars[1])
    ip_agg = pp.get_engineered_features(installments, group_vars, df_name_temp, num_agg_funcs = numeric_agg_funcs)

    ip_agg = counts_ip.merge(ip_agg, on = group_vars, how = 'left')
    
    ip_agg_client = pp.agg_numeric(ip_agg, [group_vars[0]], df_name)
    
    return ip_agg_client

def get_processed_files(debug_size, silent = True):
    num_rows = debug_size if debug_size != 0 else None
    with timer("Process application_train and application_test"):
        train, test = load_train_test(nrows = num_rows, silent = silent, treat_cat_missing = True, 
                                      treat_num_missing = True, remove_duplicated_cols = True)
        subset_ids = list(train.SK_ID_CURR) + list(test.SK_ID_CURR) if debug_size != 0 else None
        if silent == False:
            print("Train df shape:", train.shape)
            print("Test df shape:", test.shape)
#    with timer("Process Bureau"):
#        bureau_agg = bureau(subset_ids, silent = silent)
#        if silent == False:
#           print("Bureau df shape:", bureau_agg.shape)
#        train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
#        test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
#        del bureau_agg
#        gc.collect()
#    with timer("Process Bureau Balance"):
#        bureau_balance_agg = bureau_balance(subset_ids, silent = silent)
#        if silent == False:
#           print("Bureau Balance df shape:", bureau_balance_agg.shape)
#        train = train.merge(bureau_balance_agg, on = 'SK_ID_CURR', how = 'left')
#        test = test.merge(bureau_balance_agg, on = 'SK_ID_CURR', how = 'left')
#        del bureau_balance_agg
#        gc.collect()
#    with timer("Process previous_applications"):
#        previous_application_agg = previous_application(subset_ids, silent = silent)
#        if silent == False:
#           print("Previous applications df shape:", previous_application_agg.shape)
#        train = train.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')
#        test = test.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')
#        del previous_application_agg
#        gc.collect()
#    with timer("Process POS-CASH balance"):
#        cash_balance_agg = cash_balance(subset_ids, silent = silent)
#        if silent == False:
#           print("Cash Balance df shape:", cash_balance_agg.shape)
#        train = train.merge(cash_balance_agg, on = 'SK_ID_CURR', how = 'left')
#        test = test.merge(cash_balance_agg, on = 'SK_ID_CURR', how = 'left')
#        del cash_balance_agg
#        gc.collect()
#    with timer("Process credit card balance"):
#        credit_balance_agg = credit_balance(subset_ids, silent = silent)
#        if silent == False:
#           print("Credit Card Balance df shape:", credit_balance_agg.shape)
#        train = train.merge(credit_balance_agg, on = 'SK_ID_CURR', how = 'left')
#        test = test.merge(credit_balance_agg, on = 'SK_ID_CURR', how = 'left')
#        del credit_balance_agg
#        gc.collect()
#    with timer("Process installments payments"):
#        installments_payments_agg = installments_payments(subset_ids, silent = silent)
#        if silent == False:
#           print("Installments Payments df shape:", installments_payments_agg.shape)
#        train = train.merge(installments_payments_agg, on = 'SK_ID_CURR', how = 'left')
#        test = test.merge(installments_payments_agg, on = 'SK_ID_CURR', how = 'left')
#        del installments_payments_agg
#        gc.collect()
        
    return train.reset_index().drop(['index'], axis = 1), test.reset_index().drop(['index'], axis = 1)