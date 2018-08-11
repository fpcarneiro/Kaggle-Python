import pandas as pd
import numpy as np
import preprocessing as pp
from sklearn.preprocessing import Binarizer, LabelEncoder

def make_submission(model, X_train, y_train, X_test, filename = 'submission.csv'):
    model.fit(X_train, y_train)
    predicted = model.predict_proba(test_X)[:, 1]
    my_submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': predicted})
    my_submission.to_csv(filename, index=False)

train, test = pp.read_train_test(train_file = 'application_train.csv', test_file = 'application_test.csv')

cat_cols = pp.get_dtype_columns(train, [np.dtype(object)])
cat_cols2encode = [c for c in cat_cols if len(train[c].value_counts(dropna=False)) <= 3]

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

# NUMERICAL MISSING
print(pp.check_missing(train[pp.get_numerical_missing_cols(train)]))
print(pp.check_missing(test[pp.get_numerical_missing_cols(test)]))

num_missing_trans = pp.HandleMissingMedianTransformer()
train = num_missing_trans.fit_transform(train)
test = num_missing_trans.fit_transform(test)

train_y = train_labels
train_X = train.drop(['SK_ID_CURR'], axis=1)

ids = test[['SK_ID_CURR']]
test_X = test.drop(['SK_ID_CURR'], axis=1)

#train_X = pp.hot_encode(train_X)
#test_X = pp.hot_encode(test_X)

#train_X, test_X = train_X.align(test_X, join='outer', axis=1, fill_value = 0)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 0.0001)
log_reg.fit(train_X, train_y)
log_reg_pred = log_reg.predict_proba(test_X)[:, 1]

submit = ids
submit['TARGET'] = log_reg_pred

submit.head()
submit.to_csv('log_reg_baseline.csv', index = False)


import seaborn as sns
sns.set(style="darkgrid")
ax = sns.countplot(x="CODE_GENDER", data=train)

cols2bin_train = [c for c in train.columns if c.startswith("FLAG_") and len(train[c].value_counts(dropna=False)) == 2]
cols2bin_test = [c for c in test.columns if c.startswith("FLAG_") and len(test[c].value_counts(dropna=False)) == 2]

for c in cols2bin_train:
    binarizer = Binarizer()
    binarizer.fit(train[c].values.reshape(-1, 1))
    train[c] = binarizer.transform(train[c].values.reshape(-1, 1))
    test[c] = binarizer.transform(test[c].values.reshape(-1, 1))
    
    
    
flag_train = [c for c in train.columns if c.startswith("FLAG_")]
for c in flag_train[2:]:
    train.loc[:, c] = train[c].astype('bool')
    test.loc[:, c] = test[c].astype('bool')
    
    
for c in flag_train:
    print(c)
    print(train[c].unique())
    
    
le = LabelEncoder()
le.fit(train.FLAG_OWN_CAR)
train.FLAG_OWN_CAR = le.transform(train.FLAG_OWN_CAR)

le.fit(train.FLAG_OWN_REALTY)
train.FLAG_OWN_REALTY = le.transform(train.FLAG_OWN_REALTY)













