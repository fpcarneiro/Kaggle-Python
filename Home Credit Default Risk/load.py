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

def bureau(nrows = None, silent = True, treat_cat_missing = False, treat_num_missing = False):
    bureau = pp.read_dataset_csv(filename = "bureau.csv", nrows = nrows)

    if (treat_num_missing):
        if not silent:
            print("Treating numericals missing...")
#       bureau = pp.handle_missing_median(bureau, pp.get_numerical_missing_cols(bureau), group_by_cols = ["SK_ID_CURR"])
#       print(pp.check_missing(bureau[pp.get_numerical_missing_cols(bureau)]))
    
    if not silent:
        print("Bureau samples: {}".format(len(bureau)))
    
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
    bureau_cat_num_agg = pp.agg_categorical_numeric(bureau, df_name = "BUREAU", 
                                                    funcs = ['sum', 'mean', 'count'], group_var = ['SK_ID_CURR', 'CREDIT_ACTIVE'], 
                                                    target_numvar = numeric_cols)
#    bureau = pp.convert_types(bureau, print_info = True)
#    bureau_cat_num_agg = pp.convert_types(bureau_cat_num_agg, print_info = True)
#
    if not silent:
        print("Aggregating BUREAU by only 'SK_ID_CURR'...")    
    bureau_agg = pp.get_engineered_features(bureau.drop(['SK_ID_BUREAU'], axis=1), group_var = 'SK_ID_CURR', df_name = 'BB', num_agg_funcs = ['count', 'mean', 'median', 'sum'])
    
    duplicated_bureau_agg = pp.duplicate_columns(bureau_agg, verbose = not silent, progress = False)
    if not silent:
        print("Removing duplicated columns {}".format(duplicated_bureau_agg))
    if len(duplicated_bureau_agg) > 0:
        bureau_agg.drop(list(duplicated_bureau_agg.keys()), axis=1, inplace = True)
#        
#    bureau_agg = pp.convert_types(bureau_agg, print_info = True)
#    
#    train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
#    test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
#    
    duplicated_bureau_cat_num_agg = pp.duplicate_columns(bureau_cat_num_agg, verbose = not silent, progress = False)
    if not silent:
        print("Removing duplicated columns {}".format(duplicated_bureau_cat_num_agg))
    if len(duplicated_bureau_cat_num_agg) > 0:
        bureau_cat_num_agg.drop(list(duplicated_bureau_cat_num_agg.keys()), axis=1, inplace = True)
        
    return bureau_agg.merge(bureau_cat_num_agg, on = 'SK_ID_CURR', how = 'left')
#    
#    train = train.merge(bureau_cat_num_agg, on = 'SK_ID_CURR', how = 'left')
#    test = test.merge(bureau_cat_num_agg, on = 'SK_ID_CURR', how = 'left')
#    
#    bureau_agg_id_columns = list(set([c for c in bureau_agg.columns if c.startswith("SK_ID_")] + [c for c in bureau_cat_num_agg.columns if c.startswith("SK_ID_")]))
#    bureau_agg_columns = list(set([c for c in bureau_agg.columns if c not in bureau_agg_id_columns] + [c for c in bureau_cat_num_agg.columns if c not in bureau_agg_id_columns]))
#    
#    del bureau_agg, bureau_ct_table, bureau_cc_table, s_bureau_ct, s_bureau_cc, bureau_ca_table, s_bureau_ca
#    del bureau_cat_num_agg, numeric_cols
#    del duplicated_bureau_agg, duplicated_bureau_cat_num_agg
#    gc.collect()
    
    #return bureau_cat_num_agg