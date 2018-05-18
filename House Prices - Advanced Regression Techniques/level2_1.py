import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

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

X_train = train.select_dtypes(exclude=['object']).drop('SalePrice', axis=1)
X_test  = test.select_dtypes(exclude=['object'])

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)       

my_model = RandomForestRegressor()
my_model.fit(imputed_X_train, train_y)

# Use the model to make predictions
predicted_prices = my_model.predict(imputed_X_test)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission3.csv', index=False)