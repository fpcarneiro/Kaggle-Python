import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(train_X, test_X, train_y, test_y):
    forest_model = RandomForestRegressor()
    forest_model.fit(train_X, train_y)
    preds = forest_model.predict(test_X)
    return(mean_absolute_error(test_y, preds))

# Read the data
train = pd.read_csv('input/train.csv')
# Read the test data
test = pd.read_csv('input/test.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
# For the sake of keeping the example simple, we'll use only numeric predictors. 
numeric_cols = (train.select_dtypes(exclude=['object'])).drop('SalePrice', axis=1)

cols_with_missing_train = [col for col in numeric_cols if train[col].isnull().any()]
cols_with_missing_test = [col for col in numeric_cols if test[col].isnull().any()]

cols_with_missing = list(set(cols_with_missing_train + cols_with_missing_test))

reduced_X_train = train.drop(cols_with_missing_train, axis=1)
reduced_X_test  = test.drop(cols_with_missing_train, axis=1)

# Create training predictors data
train_X = reduced_X_train.drop('SalePrice', axis=1)

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = reduced_X_test
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission2.csv', index=False)