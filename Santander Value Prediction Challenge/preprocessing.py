import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

DATADIR = "input/"

def score_sq(y, y_pred):
    return(np.sqrt(mean_squared_error(y, y_pred)))

def read_train_test(train_file = 'train.csv', test_file = None):
    train = pd.read_csv(DATADIR + train_file)
    if test_file != None:
        test = pd.read_csv(DATADIR + test_file)
        return train, test
    else:
        return train

def get_feature_groups(dataset, drop_list = []):
    mydata = dataset.drop(drop_list, axis=1)
    num_columns = list(mydata.select_dtypes(exclude=['object']).columns)
    cat_columns = list(mydata.select_dtypes(include=['object']).columns)
    return (num_columns, cat_columns)

def score_model(estimator, X, y, n_folds = 5, seed = 2018, scoring_func="neg_mean_squared_error"):
    kf = KFold(n_folds, shuffle=True, random_state = seed)
    score = -cross_val_score(estimator, X, y, scoring=scoring_func, cv = kf)
    return(score)
    
def get_validation_scores(models, X_train, y_train, folds, seed = 2018):
    scores_val_mean = []
    scores_val_std = []
    scores_val = []
    names = []
    for name, model in models:
        names.append(name)
        val_scores = np.sqrt(score_model(model, X_train, y_train, n_folds = folds, seed = seed))
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

def cross_val_score_model(estimator, X, y, n_folds = 5, scoring_func="neg_mean_squared_error", seed = 2018):
    kf = KFold(n_folds, shuffle=True, random_state = seed)
    scores = np.sqrt(-cross_val_score(estimator, X, y, scoring=scoring_func, cv = kf))
    return(scores)

def cross_validate_model(estimator, X, y, n_folds = 5, scoring_func="neg_mean_squared_error", seed = 2018, return_train_score = False, jobs = -1):
    kf = KFold(n_folds, shuffle=True, random_state = seed)
    scores = cross_validate(estimator, X, y, scoring=scoring_func, cv = kf, return_train_score = return_train_score, n_jobs = jobs)
    return(scores)
    
def get_test_score(model, X_test, y_test):
    predicted = model.predict(X_test)
    score = score_sq(y_test, predicted)
    return(score)

def get_cross_validate(medels_list, X, y, folds = 5, seed = 2018, train_score = False, jobs = -1): 
    sort_by = "Score (mean)"
    if train_score:
        results = pd.DataFrame(columns = ["Estimator", "Score (mean)", "Score (std)", "CV Scores",
                                          "Train Score (mean)", "Train Score (std)", "Train CV Scores"])
    else:
        results = pd.DataFrame(columns = ["Estimator", "Score (mean)", "Score (std)", "CV Scores"])
        
    for name, model in medels_list:
        print(name)
        scores =  cross_validate_model(model, X, y, n_folds = folds, seed = seed, return_train_score = train_score, jobs = jobs)
        test_score = np.sqrt(-scores["test_score"])
        record = {"Estimator": name,
                  "Score (mean)": test_score.mean(),
                  "Score (std)": test_score.std(),
                  "CV Scores": test_score}
        
        if train_score:
            train_score = np.sqrt(-scores["train_score"])
            record["Train Score (mean)"] = train_score.mean()
            record["Train Score (std)"] = train_score.std()
            record["Train CV Scores"] = train_score
        
        results = results.append(record, ignore_index=True)
    results.sort_values(by=[sort_by], ascending = True, inplace = True)
    return results

