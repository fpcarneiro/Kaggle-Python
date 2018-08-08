import pandas as pd
import numpy as np
import preprocessing as pp
from sklearn.preprocessing import LabelEncoder

def make_submission(model, X_train, y_train, X_test, filename = 'submission.csv'):
    model.fit(X_train, y_train)
    predicted = model.predict_proba(test_X)[:, 1]
    my_submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': predicted})
    my_submission.to_csv(filename, index=False)

train, test = pp.read_train_test(train_file = 'application_train.csv', test_file = 'application_test.csv')

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

train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243
train["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243
test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

num_missing_trans = pp.HandleMissingMedianTransformer()
train = num_missing_trans.fit_transform(train)
test = num_missing_trans.fit_transform(test)

train = pp.get_domain_knowledge_features(train)
test = pp.get_domain_knowledge_features(test)

train_y = train['TARGET']
train_X = train.drop(['SK_ID_CURR', 'TARGET'], axis=1)

ids = test[['SK_ID_CURR']]
test_X = test.drop(['SK_ID_CURR'], axis=1)

# Scale each feature to 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

#train_X = pp.hot_encode(train_X)
#test_X = pp.hot_encode(test_X)

#train_X, test_X = train_X.align(test_X, join='outer', axis=1, fill_value = 0)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 0.0001)
log_reg.fit(train_X, train_y)
log_reg_pred = log_reg.predict_proba(test_X)[:, 1]

# Feature names
features = list(train.drop(['SK_ID_CURR', 'TARGET'], axis=1).columns)

submit = ids
submit['TARGET'] = log_reg_pred

submit.head()
submit.to_csv('log_reg_baseline.csv', index = False)


from sklearn.ensemble import RandomForestClassifier
# Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

# Train on the training data
random_forest.fit(train_X, train_y)
# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# Make predictions on the test data
predictions = random_forest.predict_proba(test_X)[:, 1]

submit['TARGET'] = predictions

submit.head(100)
submit.to_csv('rf_baseline.csv', index = False)



import lightgbm as lgb
from sklearn.model_selection import KFold
# Create the kfold object
k_fold = KFold(n_splits = 10, shuffle = True, random_state = 50)

for train_indices, valid_indices in k_fold.split(features):
    train_features, train_labels = train_X[train_indices], train_y[train_indices]
    valid_features, valid_labels = train_X[valid_indices], train_y[valid_indices]
    
    model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
    
    model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = "",
                  early_stopping_rounds = 100, verbose = 200)
    
    best_iteration = model.best_iteration_
    
model.fit(train_X, train_y, eval_metric = 'auc', verbose = 200)