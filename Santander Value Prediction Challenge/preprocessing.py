import pandas as pd
import numpy as np
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
    
def get_validation_scores(models, X_train, y_train, folds):
    scores_val_mean = []
    scores_val_std = []
    scores_val = []
    names = []
    for name, model in models:
        names.append(name)
        val_scores = np.sqrt(score_model(model, X_train, y_train, n_folds = folds))
        scores_val.append(val_scores)
        scores_val_mean.append(val_scores.mean())
        scores_val_std.append(val_scores.std())
        tab = pd.DataFrame({ "Model" : names, "Cross Validation (Mean)" : scores_val_mean, "Cross Validation (Std)" : scores_val_std, "Cross Validation (Scores)" : scores_val })
        tab.sort_values(by=['Cross Validation (Mean)'], ascending = True, inplace = True)
    return(tab)

def make_submission(model, X_train, y_train, X_test, ids, filename = 'submission.csv'):
    model.fit(X_train, y_train)
    predicted = np.expm1(model.predict(X_test))
    my_submission = pd.DataFrame({'ID': ids, 'target': predicted})
    my_submission.to_csv(filename, index=False)

