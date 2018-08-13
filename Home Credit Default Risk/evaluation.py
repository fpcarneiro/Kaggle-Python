import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, cross_validate

def cross_validate_model(estimator, X, y, n_folds = 5, repetitions = 1, scoring_func="roc_auc", 
                         seed = 2018, return_train_score = False, jobs = -1):
    if repetitions == 1:
        kf = KFold(n_splits = n_folds, shuffle=True, random_state = seed)
    else:
        kf = RepeatedKFold(n_splits = n_folds, n_repeats = repetitions, random_state = seed)
    scores = cross_validate(estimator, X, y, scoring=scoring_func, cv = kf, return_train_score = return_train_score, n_jobs = jobs)
    return(scores)
    
model_name = "Estimator"
score_mean = "Score (mean)"
score_std = "Score (std)"
scores_field = "CV Scores"

def get_cross_validate(medels_list, X, y, folds = 5, repetitions = 1, seed = 2018, train_score = False, jobs = -1): 
    sort_by = "Score (mean)"
    if train_score:
        results = pd.DataFrame(columns = [model_name, score_mean, score_std, scores_field,
                                          "Train Score (mean)", "Train Score (std)", "Train CV Scores"])
    else:
        results = pd.DataFrame(columns = [model_name, score_mean, score_std, scores_field])
        
    for name, model in medels_list:
        scores =  cross_validate_model(model, X, y, n_folds = folds, repetitions = repetitions, seed = seed, 
                                       return_train_score = train_score, jobs = jobs)
        test_score = scores["test_score"]
        record = {model_name: name,
                  score_mean: test_score.mean(),
                  score_std: test_score.std(),
                  scores_field: test_score}
        
        if train_score:
            train_score = scores["train_score"]
            record["Train Score (mean)"] = train_score.mean()
            record["Train Score (std)"] = train_score.std()
            record["Train CV Scores"] = train_score
        
        results = results.append(record, ignore_index=True)
    results.sort_values(by=[sort_by], ascending = True, inplace = True)
    return results