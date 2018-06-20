import preprocessing as pp
import transformers as tr
import ensemble as em
import cv_lab as cvl
from sklearn.pipeline import Pipeline
import feature_selection as fs
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, Ridge, SGDRegressor, LassoLars
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, mutual_info_regression

def get_validation_scores(models, X_train, y_train, folds, X_test = [], y_test = []):
    scores_val_mean = []
    scores_val_std = []
    scores_val = []
    scores_test = []
    names = []
    for name, model in models:
        names.append(name)
        val_scores = np.sqrt(pp.score_model(model, X_train, y_train, n_folds = folds))
        scores_val.append(val_scores)
        scores_val_mean.append(val_scores.mean())
        scores_val_std.append(val_scores.std())
        if len(X_test) != 0:
            model.fit(X_train, y_train)
            st = get_test_scores(model, X_test, y_test)
            scores_test.append(st)
    if len(X_test) != 0:
        tab = pd.DataFrame({ "Model" : names, "Cross Validation (Mean)" : scores_val_mean, "Cross Validation (Std)" : scores_val_std, "Cross Validation (Scores)" : scores_val, "Test": scores_test })
        tab.sort_values(by=['Test'], ascending = True, inplace = True)
    else:
        tab = pd.DataFrame({ "Model" : names, "Cross Validation (Mean)" : scores_val_mean, "Cross Validation (Std)" : scores_val_std, "Cross Validation (Scores)" : scores_val })
        tab.sort_values(by=['Cross Validation (Mean)'], ascending = True, inplace = True)
    return(tab)

def get_test_scores(model, X_test, y_test):
    predicted = model.predict(X_test)
    score = cvl.score_sq(y_test, predicted)
    return(score)

def make_submission(model, X_train, y_train, X_test, filename = 'submission.csv'):
    model.fit(X_train, y_train)
    predicted = np.expm1(model.predict(X_test))
    my_submission = pd.DataFrame({'ID': ids, 'target': predicted})
    my_submission.to_csv(filename, index=False)

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

train, test = pp.read_train_test()

ids = list(test.ID)

train.drop(['ID'], axis=1, inplace = True)
test.drop(['ID'], axis=1, inplace = True)

train_y = (np.log1p(train.target)).values

train.drop(['target'], axis=1, inplace = True)

log_transformation = tr.LogTransformer(threshold = 0.50)
log_transformation.fit(train)
train = log_transformation.fit_transform(train)
test = log_transformation.transform(test)

features_variance = fs.list_features_low_variance(train, train_y, threshold = .90)

train_X = train[features_variance]
test_X = test[features_variance]

features_select_from_model, pipe_select_from_model = fs.remove_features_from_model(estimator = ExtraTreesRegressor(n_estimators=200, 
                                                                                                                   max_depth=20, max_features=0.5, n_jobs=-1, random_state=0), 
                                                          scaler = RobustScaler(), X = train_X, y = train_y)
train_X_reduced = pipe_select_from_model.transform(train_X)
test_X_reduced = pipe_select_from_model.transform(test_X)

percent = SelectPercentile(mutual_info_regression, percentile=75)
train_X_reduced = percent.fit_transform(train_X_reduced, train_y)
test_X_reduced = percent.transform(test_X_reduced)

X_train, X_test, y_train, y_test = train_test_split(train_X_reduced, train_y, test_size=0.1)

##################
model_lasso = Lasso(alpha=0.0004, random_state = 1)
model_ridge = Ridge(alpha=10.0)
model_svr = SVR(C = 15, epsilon = 0.009, gamma = 0.0004, kernel = 'rbf')
model_ENet = ElasticNet()
model_KRR = KernelRidge(alpha=0.5, kernel='polynomial', degree=2, coef0=2.5)
model_byr = BayesianRidge()


model_lsvr = LinearSVR()
model_sgd = SGDRegressor()


model_rforest = RandomForestRegressor(n_estimators = 150, 
                                      max_features = 0.8,
                                      random_state=0,
                                      max_depth=15)

model_xgb = XGBRegressor(n_estimators = 200, 
                         learning_rate=0.05)

model_GBoost = GradientBoostingRegressor(n_estimators=200, 
                                         learning_rate=0.05)

model_lgb = lgb.LGBMRegressor(objective='regression',
                              metric="rmse",
                              n_estimators = 200,
                              learning_rate=0.05
                              )
model_extra = ExtraTreesRegressor(n_estimators = 200, 
                                  max_features=0.5)

#Linear Models
tree_models = []
tree_models.append(("rforest", model_rforest))
tree_models.append(("xgb", model_xgb))
tree_models.append(("GBoost", model_GBoost))
tree_models.append(("lgb", model_lgb))
tree_models.append(("extra", model_extra))

linear_models = []
linear_models.append(("enet", model_ENet))
linear_models.append(("enet", model_byr))

cross_val_table = get_validation_scores(tree_models, X_train, y_train, 5, X_test, y_test)
print(cross_val_table)

cross_val_table = get_validation_scores(tree_models, train_X_reduced, train_y, 5)
print(cross_val_table)

cross_val_table_linear = get_validation_scores(linear_models, train_X_reduced, train_y, 5)
print(cross_val_table_linear)

averaged_models = em.AveragingModels(models = [model_lgb, model_byr, model_svr, model_ridge])
stacked_averaged_models = em.StackingAveragedModels(base_models = [model_svr, model_lgb], meta_model = model_KRR)
averaged_plus = em.AveragingModels(models = [averaged_models, model_GBoost, model_xgb], weights = [0.7, 0.2, 0.1])
averaged_plus_plus = em.AveragingModels(models = [stacked_averaged_models, model_GBoost, model_xgb], weights = [0.7, 0.2, 0.1])

ensemble_models = []
ensemble_models.append(("averaged", averaged_models))
ensemble_models.append(("stacked", stacked_averaged_models))
ensemble_models.append(("averaged_plus", averaged_plus))
ensemble_models.append(("averaged_plus_plus", averaged_plus_plus))

cross_val_table_ensemble = get_validation_scores(ensemble_models, X_train, y_train, X_test, y_test)
print(cross_val_table_ensemble)

cross_val_table_ensemble = get_validation_scores(ensemble_models, train_X_reduced, train_y)
print(cross_val_table_ensemble)

make_submission(model_rforest, train_X_reduced, train_y, test_X_reduced, filename = 'submission.csv')



hyperparameters_rforest = {"n_estimators" : [150, 200, 250, 300, 400, 450, 500], "warm_start": [True, False] }
model_rforest = RandomForestRegressor(max_features = 0.8,
                                      random_state=0,
                                      max_depth=15,
                                      min_samples_leaf=5,
                                      verbose = 1)
hpg = cvl.HousePricesGridCV(model_rforest, hyperparameters = hyperparameters_rforest, n_folds = 5)
hpg.fit(train_X_reduced, train_y)
hpg.get_best_results()
hpg.plot_scores()