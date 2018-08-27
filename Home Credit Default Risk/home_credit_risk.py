import pandas as pd
import numpy as np
import preprocessing as pp
import evaluation as ev
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

# Suppress warnings from pandas
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

import gc

import feature_selection as fs

plt.style.use('fivethirtyeight')

train, test = pp.read_train_test(train_file = 'application_train.csv', test_file = 'application_test.csv')

train = train[train['CODE_GENDER'] != 'XNA']

train.loc[:, 'HOUR_APPR_PROCESS_START'] = train.loc[:, 'HOUR_APPR_PROCESS_START'].astype('object')
test.loc[:, 'HOUR_APPR_PROCESS_START'] = test.loc[:, 'HOUR_APPR_PROCESS_START'].astype('object')

# Decrease number of categories in ORGANIZATION_TYPE
train_ot_table = pp.check_categorical_cols_values(train, col = "ORGANIZATION_TYPE")
s_train = set(train_ot_table[train_ot_table.loc[:, "% of Total"] < 1].index)

test_ot_table = pp.check_categorical_cols_values(test, col = "ORGANIZATION_TYPE")
s_test = set(test_ot_table[test_ot_table.loc[:, "% of Total"] < 1].index)

l_union = list(s_train.union(s_test))

train.loc[train.ORGANIZATION_TYPE.isin(l_union), 'ORGANIZATION_TYPE'] = "Other 2"
test.loc[test.ORGANIZATION_TYPE.isin(l_union), 'ORGANIZATION_TYPE'] = "Other 2"

gc.enable()
del l_union, train_ot_table, test_ot_table
gc.collect()

train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243
train["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243
test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

cat_cols = pp.get_dtype_columns(train, [np.dtype(object)])
cat_cols2encode = [c for c in cat_cols if len(train[c].value_counts(dropna=False)) <= 2]

le = LabelEncoder()
for col in cat_cols2encode:
    le.fit(train[col])
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# CATEGORICAL MISSING
print(pp.check_missing(train[pp.get_categorical_missing_cols(train)]))
print(pp.check_missing(test[pp.get_categorical_missing_cols(test)]))

train.NAME_TYPE_SUITE.fillna("Unaccompanied", inplace= True)
test.NAME_TYPE_SUITE.fillna("Unaccompanied", inplace= True)

# High density missing categorical columns - deserves a column when performing get_dummies
# FONDKAPREMONT_MODE, WALLSMATERIAL_MODE, HOUSETYPE_MODE, EMERGENCYSTATE_MODE, OCCUPATION_TYPE

train = pd.get_dummies(train, dummy_na = True)
test = pd.get_dummies(test, dummy_na = True)

train_labels = train['TARGET']
train, test = train.align(test, join = 'inner', axis = 1)
train['TARGET'] = train_labels

# NUMERICAL MISSING
print(pp.check_missing(train[pp.get_numerical_missing_cols(train)]))
print(pp.check_missing(test[pp.get_numerical_missing_cols(test)]))

num_missing_trans = pp.HandleMissingMedianTransformer()
train = num_missing_trans.fit_transform(train)
test = num_missing_trans.fit_transform(test)

train = pp.convert_types(train, print_info = True)
test = pp.convert_types(test, print_info = True)

# FEATURE ENGINEERING
train = pp.get_domain_knowledge_features(train)
test = pp.get_domain_knowledge_features(test)

duplicated_train = pp.duplicate_columns(train, verbose = True, progress = False)
train.drop(list(duplicated_train.keys()), axis=1, inplace = True)
test.drop(list(duplicated_train.keys()), axis=1, inplace = True)

target_column = "TARGET"
application_id_columns = [c for c in test.columns if c.startswith("SK_ID_")]
application_columns = [c for c in test.columns if c not in application_id_columns]

del le, col, cat_cols, cat_cols2encode, num_missing_trans, train_labels
del duplicated_train
gc.collect()


train.to_csv("input/train_engineered_1.csv", compression="zip")
test.to_csv("input/test_engineered_1.csv", compression="zip")


bureau = pp.read_dataset_csv(filename = "bureau.csv")
print(pp.check_missing(bureau[pp.get_numerical_missing_cols(bureau)]))
bureau = pp.handle_missing_median(bureau, pp.get_numerical_missing_cols(bureau), group_by_cols = ["SK_ID_CURR"])
print(pp.check_missing(bureau[pp.get_numerical_missing_cols(bureau)]))

bureau_ct_table = pp.check_categorical_cols_values(bureau, col = "CREDIT_TYPE")
s_bureau_ct = set(bureau_ct_table[bureau_ct_table.loc[:, "% of Total"] < 1].index)
bureau.loc[bureau.CREDIT_TYPE.isin(s_bureau_ct), 'CREDIT_TYPE'] = "Other"

bureau_cc_table = pp.check_categorical_cols_values(bureau, col = "CREDIT_CURRENCY")
s_bureau_cc = set(bureau_cc_table[bureau_cc_table.loc[:, "% of Total"] < 1].index)
bureau.loc[bureau.CREDIT_CURRENCY.isin(s_bureau_cc), 'CREDIT_CURRENCY'] = "Other"

bureau_ca_table = pp.check_categorical_cols_values(bureau, col = "CREDIT_ACTIVE")
s_bureau_ca = set(bureau_ca_table[bureau_ca_table.loc[:, "% of Total"] < 1].index)
bureau.loc[bureau.CREDIT_ACTIVE.isin(s_bureau_ca), 'CREDIT_ACTIVE'] = "Other"

bureau = pp.handle_missing_median(bureau, pp.get_numerical_missing_cols(bureau), group_by_cols = ["CREDIT_TYPE"])
print(pp.check_missing(bureau[pp.get_numerical_missing_cols(bureau)]))

#df, df_name, group_var = ['SK_ID_CURR', 'CREDIT_ACTIVE'], funcs = ['sum', 'mean'], target_numvar = ['DAYS_CREDIT', 'AMT_ANNUITY']
numeric_cols = pp.get_dtype_columns(bureau, dtypes = [np.dtype(np.int64), np.dtype(np.float64)])
bureau_cat_num_agg = pp.agg_categorical_numeric(bureau, df_name = "BUREAU", 
                                                funcs = ['sum', 'mean', 'std'], group_var = ['SK_ID_CURR', 'CREDIT_ACTIVE'], 
                                                target_numvar = numeric_cols)

bureau = pp.convert_types(bureau, print_info = True)
bureau_cat_num_agg = pp.convert_types(bureau_cat_num_agg, print_info = True)

bureau.to_csv("input/bureau_engineered.csv", compression="zip")
bureau_cat_num_agg.to_csv("input/bureau_cat_num_agg.csv", compression="zip")

bureau_agg = pp.get_engineered_features(bureau.drop(['SK_ID_BUREAU'], axis=1), group_var = 'SK_ID_CURR', df_name = 'BUREAU', num_agg_funcs = ['count', 'mean', 'median', 'sum'])

duplicated_bureau_agg = pp.duplicate_columns(bureau_agg, verbose = True, progress = False)
if len(duplicated_bureau_agg) > 0:
    bureau_agg.drop(list(duplicated_bureau_agg.keys()), axis=1, inplace = True)
    
bureau_agg = pp.convert_types(bureau_agg, print_info = True)

train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

duplicated_bureau_cat_num_agg = pp.duplicate_columns(bureau_cat_num_agg, verbose = True, progress = False)
if len(duplicated_bureau_cat_num_agg) > 0:
    bureau_cat_num_agg.drop(list(bureau_cat_num_agg.keys()), axis=1, inplace = True)

train = train.merge(bureau_cat_num_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_cat_num_agg, on = 'SK_ID_CURR', how = 'left')

bureau_agg_id_columns = list(set([c for c in bureau_agg.columns if c.startswith("SK_ID_")] + [c for c in bureau_cat_num_agg.columns if c.startswith("SK_ID_")]))
bureau_agg_columns = list(set([c for c in bureau_agg.columns if c not in bureau_agg_id_columns] + [c for c in bureau_cat_num_agg.columns if c not in bureau_agg_id_columns]))

del bureau_agg, bureau_ct_table, bureau_cc_table, s_bureau_ct, s_bureau_cc, bureau_ca_table, s_bureau_ca
del bureau_cat_num_agg, numeric_cols
del duplicated_bureau_agg, duplicated_bureau_cat_num_agg
gc.collect()

train.to_csv("input/train_engineered_2.csv", compression="zip")
test.to_csv("input/test_engineered_2.csv", compression="zip")

group_vars = ['SK_ID_BUREAU', 'SK_ID_CURR']
bureau_balance = pp.read_dataset_csv(filename = "bureau_balance.csv")
bureau_balance = pp.convert_types(bureau_balance, print_info = True)
#bureau_balance_agg = pp.aggregate_client(bureau_balance, parent_df = bureau[group_vars], group_vars = group_vars, 
#                                         df_names = ['bureau_balance', 'client'])

bureau_balance_agg = pp.get_engineered_features(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = "bureau_balance", num_agg_funcs = ['count', 'min', 'max'], cat_agg_funcs = ['sum'], cols_alias = ['count'])
cols_status = [c for c in bureau_balance_agg.columns if c.endswith("_count") and c.find("_STATUS_") != -1 and c not in ["bureau_balance_STATUS_X_count", "bureau_balance_STATUS_C_count", "bureau_balance_STATUS_0_count"]]
# DPD ==Days Past Due
bureau_balance_agg["bureau_balance_DPD_count"] = bureau_balance_agg.loc[:, cols_status].sum(axis=1)
#bureau_balance_agg["bureau_balance_DPD_PERCENT"] = bureau_balance_agg["DPD_COUNT"]/bureau_balance_agg["bureau_balance_MONTHS_BALANCE_count"]
#bureau_balance_agg[bureau_balance_agg.SK_ID_BUREAU.isin(bureau.SK_ID_BUREAU) == False]
bureau_balance_agg = bureau_balance_agg.merge(bureau[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'inner')
bureau_balance_agg = bureau_balance_agg.drop([group_vars[0]], axis=1)
bureau_balance_agg_by_client = pp.agg_numeric(bureau_balance_agg, group_var = group_vars[1], df_name = 'CLIENT')

cols_status_percent = [c for c in bureau_balance_agg_by_client.columns if c.endswith("_count_sum") and c.find("_STATUS_") != -1] + ["CLIENT_bureau_balance_DPD_count_sum"]
for c in cols_status_percent:
    bureau_balance_agg_by_client[c + "_PERCENT"] = bureau_balance_agg_by_client[c]/bureau_balance_agg_by_client["CLIENT_bureau_balance_MONTHS_BALANCE_count_sum"]

duplicated_bureau_balance_agg_by_client = pp.duplicate_columns(bureau_balance_agg_by_client, verbose = True, progress = False)
if len(duplicated_bureau_balance_agg_by_client) > 0:
    bureau_balance_agg_by_client.drop(list(duplicated_bureau_balance_agg_by_client.keys()), axis=1, inplace = True)
    
bureau_balance_agg_by_client = pp.convert_types(bureau_balance_agg_by_client, print_info = True)

train = train.merge(bureau_balance_agg_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_balance_agg_by_client, on = 'SK_ID_CURR', how = 'left')

bureau_balance_agg_id_columns = [c for c in bureau_balance_agg_by_client.columns if c.startswith("SK_ID_")]
bureau_balance_agg_columns = [c for c in bureau_balance_agg_by_client.columns if c not in bureau_balance_agg_id_columns]

gc.enable()
del bureau, bureau_balance, bureau_balance_agg, group_vars
del cols_status, bureau_balance_agg_by_client
del cols_status_percent
del c, duplicated_bureau_balance_agg_by_client
gc.collect()

train.to_csv("input/train_engineered_3.csv", compression="zip")
test.to_csv("input/test_engineered_3.csv", compression="zip")

previous_application = pp.read_dataset_csv(filename = "previous_application.csv")

print(pp.check_missing(previous_application[pp.get_numerical_missing_cols(previous_application)]))
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

# Decrease number of categories in NAME_CASH_LOAN_PURPOSE
previous_application_nclp_table = pp.check_categorical_cols_values(previous_application, col = "NAME_CASH_LOAN_PURPOSE")
s_previous_application = set(previous_application_nclp_table[previous_application_nclp_table.loc[:, "% of Total"] < 1].index)
previous_application.loc[previous_application.NAME_CASH_LOAN_PURPOSE.isin(s_previous_application), 'NAME_CASH_LOAN_PURPOSE'] = "Other 2"

previous_application_npt_table = pp.check_categorical_cols_values(previous_application, col = "NAME_PAYMENT_TYPE")
s_previous_application = set(previous_application_npt_table[previous_application_npt_table.loc[:, "% of Total"] < 1].index)
previous_application.loc[previous_application.NAME_PAYMENT_TYPE.isin(s_previous_application), 'NAME_PAYMENT_TYPE'] = "Other 2"

previous_application_crr_table = pp.check_categorical_cols_values(previous_application, col = "CODE_REJECT_REASON")
s_previous_application = set(previous_application_crr_table[previous_application_crr_table.loc[:, "% of Total"] < 1].index)
previous_application.loc[previous_application.CODE_REJECT_REASON.isin(s_previous_application), 'CODE_REJECT_REASON'] = "Other 2"

previous_application_nts_table = pp.check_categorical_cols_values(previous_application, col = "NAME_TYPE_SUITE")
s_previous_application = set(previous_application_nts_table[previous_application_nts_table.loc[:, "% of Total"] < 1.5].index)
previous_application.loc[previous_application.NAME_TYPE_SUITE.isin(s_previous_application), 'NAME_TYPE_SUITE'] = "Other 2"

previous_application_ngc_table = pp.check_categorical_cols_values(previous_application, col = "NAME_GOODS_CATEGORY")
s_previous_application = set(previous_application_ngc_table[previous_application_ngc_table.loc[:, "% of Total"] < 1].index)
previous_application.loc[previous_application.NAME_GOODS_CATEGORY.isin(s_previous_application), 'NAME_GOODS_CATEGORY'] = "Other 2"

previous_application_ct_table = pp.check_categorical_cols_values(previous_application, col = "CHANNEL_TYPE")
s_previous_application = set(previous_application_ct_table[previous_application_ct_table.loc[:, "% of Total"] < 1].index)
previous_application.loc[previous_application.CHANNEL_TYPE.isin(s_previous_application), 'CHANNEL_TYPE'] = "Other 2"

previous_application_nsi_table = pp.check_categorical_cols_values(previous_application, col = "NAME_SELLER_INDUSTRY")
s_previous_application = set(previous_application_nsi_table[previous_application_nsi_table.loc[:, "% of Total"] < 2].index)
previous_application.loc[previous_application.NAME_SELLER_INDUSTRY.isin(s_previous_application), 'NAME_SELLER_INDUSTRY'] = "Other 2"

previous_application_pc_table = pp.check_categorical_cols_values(previous_application, col = "PRODUCT_COMBINATION")
s_previous_application = set(previous_application_pc_table[previous_application_pc_table.loc[:, "% of Total"] < 1].index)
previous_application.loc[previous_application.PRODUCT_COMBINATION.isin(s_previous_application), 'PRODUCT_COMBINATION'] = "Other 2"

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

#previous_application['APP_CREDIT_PERC'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']

previous_application = pp.handle_missing_median(previous_application, pp.get_numerical_missing_cols(previous_application), group_by_cols = ["NAME_CONTRACT_STATUS"])
print(pp.check_missing(previous_application[pp.get_numerical_missing_cols(previous_application)]))

numeric_cols = pp.get_dtype_columns(previous_application, dtypes = [np.dtype(np.int64), np.dtype(np.float64)])
previous_application_cat_num_agg = pp.agg_categorical_numeric(previous_application, df_name = "previous_application", 
                                                funcs = ['sum', 'mean', 'std'], group_var = ['SK_ID_CURR', 'NAME_CONTRACT_STATUS'], 
                                                target_numvar = numeric_cols)

previous_application_agg = pp.get_engineered_features(previous_application.drop(['SK_ID_PREV'], axis=1), group_var = 'SK_ID_CURR', df_name = 'previous', num_agg_funcs = ['count', 'mean', 'median', 'sum'])

duplicated_previous_application_agg = pp.duplicate_columns(previous_application_agg, verbose = True, progress = False)
if len(duplicated_previous_application_agg) > 0:
    previous_application_agg.drop(list(duplicated_previous_application_agg.keys()), axis=1, inplace = True)
    
duplicated_previous_application_cat_num_agg = pp.duplicate_columns(previous_application_cat_num_agg, verbose = True, progress = False)
if len(duplicated_previous_application_cat_num_agg) > 0:
    previous_application_cat_num_agg.drop(list(duplicated_previous_application_cat_num_agg.keys()), axis=1, inplace = True)
    
previous_application_agg_id_columns = list(set([c for c in previous_application_agg.columns if c.startswith("SK_ID_")] + [c for c in previous_application_cat_num_agg.columns if c.startswith("SK_ID_")]))
previous_application_agg_columns = list(set([c for c in previous_application_agg.columns if c not in previous_application_agg_id_columns] + [c for c in previous_application_cat_num_agg.columns if c not in previous_application_agg_id_columns]))

previous_application_agg = pp.convert_types(previous_application_agg, print_info = True)
previous_application_cat_num_agg = pp.convert_types(previous_application_cat_num_agg, print_info = True)

train = train.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')

train = train.merge(previous_application_cat_num_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(previous_application_cat_num_agg, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del previous_application, previous_application_agg
del previous_application_nclp_table, s_previous_application, previous_application_npt_table, previous_application_crr_table, previous_application_nts_table, previous_application_ngc_table, 
del previous_application_ct_table, previous_application_nsi_table, previous_application_pc_table, cat_cols2encode, numeric_cols
del previous_application_cat_num_agg, duplicated_previous_application_agg, duplicated_previous_application_cat_num_agg
gc.collect()

train.to_csv("input/train_engineered_4.csv", compression="zip")
test.to_csv("input/test_engineered_4.csv", compression="zip")

group_vars = ['SK_ID_PREV', 'SK_ID_CURR']
cash = pp.read_dataset_csv(filename = "POS_CASH_balance.csv")

cash_ncs_table = pp.check_categorical_cols_values(cash, col = "NAME_CONTRACT_STATUS")
s_cash = set(cash_ncs_table[cash_ncs_table.loc[:, "% of Total"] < 1].index)
cash.loc[cash.NAME_CONTRACT_STATUS.isin(s_cash), 'NAME_CONTRACT_STATUS'] = "Other 2"

cash = pp.convert_types(cash, print_info=True)

cash_agg = pp.get_engineered_features(cash, group_var = 'SK_ID_PREV', df_name = "CASH", num_agg_funcs = ['count', 'min', 'max'], cat_agg_funcs = ['sum'], cols_alias = ['count'])
cash_agg = cash_agg.merge(cash[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'inner')
cash_agg = cash_agg.drop([group_vars[0]], axis=1)
cash_agg_by_client = pp.agg_numeric(cash_agg, group_var = group_vars[1], df_name = 'CLIENT', agg_funcs = ['count', 'mean', 'median', 'sum'])

#cash_agg = pp.aggregate_client_2(cash, group_vars = group_vars, df_names = ['cash', 'client'])

cash_agg = pp.convert_types(cash_agg, print_info = True)
cash_agg_by_client = pp.convert_types(cash_agg_by_client, print_info = True)

duplicated_cash_agg_by_client = pp.duplicate_columns(cash_agg_by_client, verbose = True, progress = False)
if len(duplicated_cash_agg_by_client) > 0:
    cash_agg_by_client.drop(list(duplicated_cash_agg_by_client.keys()), axis=1, inplace = True)

train = train.merge(cash_agg_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(cash_agg_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del cash, cash_agg
del cash_ncs_table, s_cash
del cash_agg_by_client, duplicated_cash_agg_by_client
gc.collect()

train.to_csv("input/train_engineered_5.csv", compression="zip")
test.to_csv("input/test_engineered_5.csv", compression="zip")

credit_card_balance = pp.convert_types(pp.read_dataset_csv(filename = "credit_card_balance.csv"), print_info=True)
credit_card_balance_agg = pp.aggregate_client_2(credit_card_balance, group_vars = group_vars, df_names = ['credit', 'client'])
train = train.merge(credit_card_balance_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(credit_card_balance_agg, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del credit_card_balance, credit_card_balance_agg
gc.collect()

installments_payments = pp.convert_types(pp.read_dataset_csv(filename = "installments_payments.csv"), print_info=True)
installments_payments_agg = pp.aggregate_client_2(installments_payments, group_vars = group_vars, df_names = ['installments', 'client'])

duplicated_installments_payments_agg = pp.duplicate_columns(installments_payments_agg, verbose = True, progress = False)
if len(duplicated_installments_payments_agg) > 0:
    installments_payments_agg.drop(list(duplicated_installments_payments_agg.keys()), axis=1, inplace = True)

train = train.merge(installments_payments_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(installments_payments_agg, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del installments_payments, installments_payments_agg, group_vars
del duplicated_installments_payments_agg
gc.collect()

train_X = train.fillna(0)
test_X = test.fillna(0)

# PREPARING TO TRAIN
train_y = train_X['TARGET']
train_X = train_X.drop(['SK_ID_CURR', 'TARGET'], axis=1)

ids = test_X['SK_ID_CURR']
test_X = test_X.drop(['SK_ID_CURR'], axis=1)

duplicated = pp.duplicate_columns(train_X, verbose = True, progress = False)
if len(duplicated)>0:
    train_X.drop(list(duplicated.keys()), axis=1, inplace = True)
    test_X.drop(list(duplicated.keys()), axis=1, inplace = True)

features_variance = fs.list_features_low_variance(train_X, train_y, .98)
train_X_reduced = train_X[features_variance]
test_X_reduced = test_X[features_variance]

del train_X, test_X
gc.collect()

pipeline = Pipeline([
                     ('scaler', MinMaxScaler(feature_range = (0, 1))),
                     #('low_variance', VarianceThreshold(0.98 * (1 - 0.98))),
                     ('reduce_dim', SelectFromModel(lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=1500, objective = 'binary', 
                                   learning_rate = 0.05, silent = False,
                                   subsample = 0.8, colsample_bytree = 0.5))),
                     ])

pipeline.fit(train_X_reduced, train_y)

features_select_from_model = list(train_X_reduced.loc[:, pipeline.named_steps['reduce_dim'].get_support()].columns)

train_X_reduced = pipeline.transform(train_X_reduced)
test_X_reduced = pipeline.transform(test_X_reduced)

###############################################################################
#XGBOOST
###############################################################################

xgb_train = xgb.DMatrix(data=train_X_reduced, label=train_y, feature_names = features_select_from_model)
xg_test = xgb.DMatrix(data=test_X_reduced, feature_names = features_select_from_model)

xgb_params = dict()
xgb_params["booster"] = "gbtree"
xgb_params["objective"] = "binary:logistic"
xgb_params["colsample_bytree"] = 0.4385
xgb_params["subsample"] = 0.7379
xgb_params["max_depth"] = 3
xgb_params['reg_alpha'] = 0.1
xgb_params['reg_lambda'] = 0.1
xgb_params["learning_rate"] = 0.09
xgb_params["min_child_weight"] = 2

xgb_results = xgb.cv(dtrain=xgb_train, params=xgb_params, nfold=3,
                    num_boost_round=1500, early_stopping_rounds=50, metrics="auc", as_pandas=True, seed=2018, verbose_eval = 10)
#xgb_results.head()
#print((xgb_results["test-auc-mean"]).tail(1))

xgbooster = xgb.train(params = xgb_params, dtrain = xgb_train, num_boost_round = 750, maximize = True)

#import matplotlib.pyplot as plt

#xgb.plot_tree(xgbooster,num_trees=0)
#plt.rcParams['figure.figsize'] = [1000, 1000]
#plt.show()

#xgb.plot_importance(xgbooster)
#plt.rcParams['figure.figsize'] = [50, 50]
#plt.show()

pred = xgbooster.predict(xg_test)
my_submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': pred})
my_submission.to_csv("xgb_dmatrix.csv", index=False)

# LIGHT GBM
lgb_train = lgb.Dataset(train_X_reduced, label=train_y, feature_name = features_select_from_model)
#lgb_test = lgb.Dataset(test_X_reduced)

lgb_params = {}
lgb_params['boosting_type'] = 'gbdt'
lgb_params['objective'] = 'binary'
lgb_params['learning_rate'] = 0.0596
lgb_params['reg_alpha'] = 0.1
lgb_params['reg_lambda'] = 0.1
lgb_params['max_depth'] = 3
lgb_params['subsample'] = 0.7379
lgb_params["colsample_bytree"] = 0.4385
lgb_params['metric'] = 'auc'

# Params to test later: stratified, shuffle, 
lgb_results = lgb.cv(train_set = lgb_train, params = lgb_params, num_boost_round = 1500, nfold = 3,
       metrics='auc', early_stopping_rounds = 50, verbose_eval = 10, seed=2018)

lgb_booster = lgb.train(params = lgb_params, train_set = lgb_train, num_boost_round = 1450)

lgb_predict = lgb_booster.predict(test_X_reduced)
my_submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': lgb_predict})
my_submission.to_csv("lgb_dataset.csv", index=False)
#[1260]  cv_agg's auc: 0.780272 + 0.00120376

lgb_results = lgb.cv(train_set = lgb_train, params = lgb_grid.best_params_, num_boost_round = 1500, nfold = 3,
       metrics='auc', early_stopping_rounds = 50, verbose_eval = 10, seed=2018)

lgb_booster = lgb.train(params = lgb_grid.best_params_, train_set = lgb_train, num_boost_round = 830)









importances_tree = fs.get_feature_importance(lgb.LGBMClassifier(n_estimators=1500, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = 1, random_state = 50), train_X, train_y)
fs.plot_features_importances(importances_tree, show_importance_zero = False)

def go_cv(trainset_X, trainset_y):
    #model_gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=0.05, max_depth=5, subsample = 0.8, random_state=0)
    #model_logc = LogisticRegression(C = 0.0001)
    #model_rf = RandomForestClassifier(n_estimators = 10, n_jobs = 1)
    model_lgb = lgb.LGBMClassifier(n_estimators=1500, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = 1, random_state = 50)
    model_xgb = xgb.XGBClassifier(colsample_bytree=0.35, gamma=0.027, 
                             learning_rate=0.03, max_depth=4, 
                             min_child_weight=1.7817, n_estimators=1500,
                             reg_alpha=0.43, reg_lambda=0.88,
                             subsample=0.5213, silent=1,
                             random_state = 0, n_jobs = 1)

    models = []
    #models.append(("lr", model_logc))
    #models.append(("gb", model_gbc))
    models.append(("lgb", model_lgb))
    #models.append(("rf", model_rf))
    models.append(("xgb", model_xgb))

    seed = 2018
    results = ev.get_cross_validate(models, trainset_X, trainset_y, 
                                       folds = 3, repetitions = 1, seed = seed, train_score = False)
    return results

def submit(model, ids, testset_X, filename = 'submission.csv'):
    predicted = model.predict_proba(testset_X)[:, 1]
    my_submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': predicted})
    my_submission.to_csv(filename, index=False)
    
    
train_cv = go_cv(train_X, train_y)

model_xgb.fit(xgtrain)

submit(model_xgb, ids, test_X, filename = 'submission_xgb.csv')



#dtrain = xgb.DMatrix(train[featureNames].values, label=train['target'].values)


params = {
         'gamma' : 0.027, 
         'learning_rate' : 0.03,
         'max_depth' : 4,
         'min_child_weight' : 1.7817,
         'n_estimators' : 1500,
         'reg_alpha' : 0.43,
         'reg_lambda' : 0.88,
         'subsample' : 0.5213,
         'silent' : 1,
         'n_jobs' : 1,
         'objective':'binary:logistic', 
         'eval_metric': 'auc'}

clf = xgb.train(params, xgtrain, 2000)

pred = clf.predict(xgtest)
my_submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': pred})
my_submission.to_csv("xgb_dmatrix.csv", index=False)




























lgb.LGBMClassifier()



# GRID SEARCH
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

xgb_params_fixed = {"objective" : "binary:logistic", "silent" : False}

xgb_params_distribution = {"max_depth" : [1,3,5],
                      "n_estimators" : [100],
                      "learning_rate": uniform(),
                      "subsample": [0.5],
                      "colsample_bytree" : [0.5]}

lgb_params_fixed = {"objective" : "binary", "silent" : False}

lgb_params_distribution = {"max_depth" : [3,5],
                      "n_estimators" : randint(400, 1000),
                      "learning_rate": [0.0596],
                      "subsample": [0.7379],
                      "colsample_bytree" : [0.438]}

lgb_grid = RandomizedSearchCV(estimator = lgb.LGBMClassifier(**lgb_params_fixed, seed = 123), param_distributions=lgb_params_distribution, n_iter = 5, cv = 3, scoring = "roc_auc", random_state = 123)
lgb_grid.fit(train_X_reduced, train_y)


colsample_bytree=0.4385722446796244, learning_rate=0.05967789660956835, max_depth=3, n_estimators=408, subsample=0.7379954057320357, score=0.7781033030827368, total= 1.2min