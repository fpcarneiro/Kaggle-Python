import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

DATADIR = "input/"

def read_train_test(train_file = 'train.csv', test_file = None):
    train = pd.read_csv(DATADIR + train_file)
    if test_file != None:
        test = pd.read_csv(DATADIR + test_file)
        return train, test
    else:
        return train

def get_feature_groups(dataset, drop_list = ['dataset', 'Id']):
    mydata = dataset.drop(drop_list, axis=1)
    num_columns = list(mydata.select_dtypes(exclude=['object']).columns)
    cat_columns = list(mydata.select_dtypes(include=['object']).columns)
    return (num_columns, cat_columns)

def score_model(estimator, X, y, n_folds = 5, scoring_func="neg_mean_squared_error"):
    kf = KFold(n_folds, shuffle=True)
    score = -cross_val_score(estimator, X, y, scoring=scoring_func, cv = kf)
    return(score)

