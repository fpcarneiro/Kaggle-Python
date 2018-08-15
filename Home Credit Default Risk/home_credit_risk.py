import pandas as pd
import numpy as np
import preprocessing as pp
import evaluation as ev
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

# Suppress warnings from pandas
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

import gc

plt.style.use('fivethirtyeight')

train, test = pp.read_train_test(train_file = 'application_train.csv', test_file = 'application_test.csv')

train = train[train['CODE_GENDER'] != 'XNA']

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

del le, col, cat_cols, cat_cols2encode, num_missing_trans, train_labels
gc.collect()

bureau = pp.read_dataset_csv(filename = "bureau.csv")
bureau_agg = pp.get_engineered_features(bureau.drop(['SK_ID_BUREAU'], axis=1), group_var = 'SK_ID_CURR', df_name = 'bureau')
train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

#del bureau, bureau_agg
#gc.collect()
group_vars = ['SK_ID_BUREAU', 'SK_ID_CURR']
bureau_balance = pp.read_dataset_csv(filename = "bureau_balance.csv")
bureau_balance_agg = pp.aggregate_client(bureau_balance, bureau[group_vars], group_vars = group_vars, 
                                         df_names = ['bureau_balance', 'client'])
train = train.merge(bureau_balance_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_balance_agg, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del bureau, bureau_agg, bureau_balance, bureau_balance_agg, group_vars
gc.collect()

previous_application = pp.read_dataset_csv(filename = "previous_application.csv")
previous_application.drop(['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED'], axis=1, inplace = True)
previous_application_agg = pp.get_engineered_features(previous_application.drop(['SK_ID_PREV'], axis=1), group_var = 'SK_ID_CURR', df_name = 'previous')
train = train.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del previous_application, previous_application_agg
gc.collect()

installments_payments = pp.read_dataset_csv(filename = "installments_payments.csv")

previous_application = pp.read_dataset_csv(file = "previous_application.csv")

previous_application.drop(['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED'], axis=1, inplace = True)
previous_application = pp.handle_missing_median(previous_application, pp.get_numerical_missing_cols(previous_application))

installments_payments = pp.read_dataset_csv(file = "installments_payments.csv")

#bureau_agg = pp.get_engineered_features(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
#bureau_balance_agg = pp.get_engineered_features(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')

# FEATURE ENGINEERING
train = pp.get_domain_knowledge_features(train)
test = pp.get_domain_knowledge_features(test)

bureau_agg, bureau_balance_agg = pp.features(bureau, bureau_balance)
previous_application_agg = pp.get_engineered_features(previous_application, group_var = 'SK_ID_CURR', df_name = 'previous')


original_features = list(train.columns)
print('Original Number of Features: ', len(original_features))

# Merge with the value counts of bureau

#train = train.merge(bureau_agg, left_on = 'SK_ID_CURR', right_index = True, how = 'left')

# Merge with the monthly information grouped by client
train = train.merge(bureau_balance_agg, on = 'SK_ID_CURR', how = 'left')

new_features = list(train.columns)
print('Number of features using previous loans from other institutions data: ', len(new_features))

# Merge with the value counts of bureau

#test = test.merge(bureau_counts, left_on = 'SK_ID_CURR', right_index = True, how = 'left')

# Merge with the value counts of bureau balance
test = test.merge(bureau_balance_agg, on = 'SK_ID_CURR', how = 'left')

print('Shape of Testing Data: ', test.shape)

train.fillna(0, inplace= True)
test.fillna(0, inplace= True)

# PREPARING TO TRAIN
train_y = train['TARGET']
train_X = train.drop(['SK_ID_CURR', 'TARGET'], axis=1)

ids = test['SK_ID_CURR']
test_X = test.drop(['SK_ID_CURR'], axis=1)

duplicated = pp.duplicate_columns(train_X)
train_X.drop(list(duplicated.keys()), axis=1, inplace = True)
test_X.drop(list(duplicated.keys()), axis=1, inplace = True)

pipeline = Pipeline([
                     ('scaler', MinMaxScaler(feature_range = (0, 1))),
                     ('low_variance', VarianceThreshold()),
                     ('reduce_dim', SelectFromModel(LogisticRegression(C = 0.0001))),
                     ])

pipeline.fit(train_X, train_y)
train_X = pipeline.transform(train_X)
test_X = pipeline.transform(test_X)

def go_cv(trainset_X, trainset_y):
    model_gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=0.05, max_depth=5, subsample = 0.8, random_state=0)
    model_logc = LogisticRegression(C = 0.0001)
    model_rf = RandomForestClassifier(n_estimators = 10)
    model_lgb = lgb.LGBMClassifier(n_estimators=10, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
    model_xgb = XGBClassifier(colsample_bytree=0.35, gamma=0.027, 
                             learning_rate=0.03, max_depth=4, 
                             min_child_weight=1.7817, n_estimators=10,
                             reg_alpha=0.43, reg_lambda=0.88,
                             subsample=0.5213, silent=1,
                             random_state = 0)

    models = []
    models.append(("lr", model_logc))
    models.append(("gb", model_gbc))
    models.append(("lgb", model_lgb))
    models.append(("rf", model_rf))
    models.append(("xgb", model_xgb))

    seed = 2018
    results = ev.get_cross_validate(models, trainset_X, trainset_y, 
                                       folds = 3, repetitions = 1, seed = seed, train_score = False)
    return results

def submit(model, trainset_X, trainset_y, ids, testset_X, filename = 'submission.csv'):
    model.fit(trainset_X, trainset_y)
    predicted = model.predict_proba(testset_X)[:, 1]
    my_submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': predicted})
    my_submission.to_csv(filename, index=False)
    
    
train_cv = go_cv(train_X, train_y)
submit(model_xgb, train_X, train_y, ids, test_X, filename = 'submission_xgb.csv')





































from sklearn.model_selection import KFold
# Create the kfold object
k_fold = KFold(n_splits = 10, shuffle = True, random_state = 50)

for train_indices, valid_indices in k_fold.split(features):
    train_features, train_labels = train_X[train_indices], train_y[train_indices]
    valid_features, valid_labels = train_X[valid_indices], train_y[valid_indices]
    
    model = lgb.LGBMClassifier(n_estimators=1500, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
    
    model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = "",
                  early_stopping_rounds = 100, verbose = 200)
    
    best_iteration = model.best_iteration_
    
