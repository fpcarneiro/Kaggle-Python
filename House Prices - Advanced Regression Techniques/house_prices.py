import preprocessing as pp
import transformers as tr
import ensemble as em
from cv_lab import score_sq
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

def get_validation_scores(models, X_train, y_train, X_test = [], y_test = []):
    scores_val_mean = []
    scores_val_std = []
    scores_val = []
    scores_test = []
    names = []
    for name, model in models:
        names.append(name)
        val_scores = np.sqrt(pp.score_model(model, X_train, y_train, n_folds = 10))
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
    score = score_sq(y_test, predicted)
    return(score)

def make_submission(model, X_train, y_train, X_test, filename = 'submission.csv'):
    model.fit(X_train, y_train)
    predicted = np.expm1(model.predict(X_test))
    my_submission = pd.DataFrame({'Id': ids, 'SalePrice': predicted})
    my_submission.to_csv(filename, index=False)

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

train, test = pp.read_train_test()

ids = list(test.Id)

train = pp.drop_outliers(train)

train.drop(['Id'], axis=1, inplace = True)
test.drop(['Id'], axis=1, inplace = True)

train_y = (np.log1p(train.SalePrice)).values

train.drop(['SalePrice'], axis=1, inplace = True)

basic_pipeline = Pipeline([('convert', tr.Numeric2CategoryTransformer(["MSSubClass", "MoSold"])),
                 ('missing', tr.HandleMissingTransformer()),
                 #('encode_features', tr.EncodeTransformer()),
                 #('date_related_features', tr.DateRelatedFeaturesTransformer()),
                 #('neighbourhood_features', tr.NeighbourhoodRelatedFeaturesTransformer()),
                 ])

train = basic_pipeline.fit_transform(train)
test = basic_pipeline.fit_transform(test)

log_transformation = tr.LogTransformer(threshold = 0.75)
log_transformation.fit(train)
train = log_transformation.fit_transform(train)
test = log_transformation.transform(test)

second_pipeline = Pipeline([
                 #('have_stuff_features', tr.HaveStuffTransformer()),
                 ('hot_encode', tr.HotEncodeTransformer()),
                 ])

train = second_pipeline.fit_transform(train)
test = second_pipeline.fit_transform(test)

train, test = train.align(test,join='outer', axis=1, fill_value = 0)

features_variance = fs.list_features_low_variance(train, train_y, .98)

train_X = train[features_variance]
test_X = test[features_variance]

importances = fs.get_feature_importance(Lasso(alpha=0.0005), train_X, train_y)
fs.plot_features_importances(importances, show_importance_zero = False)

features_select_from_model, pipe_select_from_model = fs.remove_features_from_model(estimator = Lasso(alpha=0.0005), 
                                                          scaler = RobustScaler(), X = train_X, y = train_y)
train_X_reduced = pipe_select_from_model.transform(train_X)
test_X_reduced = pipe_select_from_model.transform(test_X)

X_train, X_test, y_train, y_test = train_test_split(train_X_reduced, train_y, test_size=0.2)

##################
seed = 2018

model_ridge = Ridge(alpha=12.0, random_state=seed)
model_KRR = KernelRidge(alpha=0.7, kernel='polynomial', degree=2, coef0=2.0, gamma=0.00366)
model_svr = SVR(C=5.22, epsilon = 0.0774, gamma = 0.0004, kernel = 'rbf')
model_byr = BayesianRidge()
model_ENet = ElasticNet(alpha=0.0001, l1_ratio=0.551, random_state=seed, max_iter = 10000)
model_lasso = Lasso(alpha=0.0004, random_state = seed)
model_lsvr = LinearSVR(C=0.525, epsilon= 0.04, random_state=seed)
model_lasso_lars = LassoLars(alpha=1.22e-05)

model_rforest = RandomForestRegressor(n_estimators = 300, max_features = 0.4, 
                                      min_samples_split = 4,
                                      random_state=seed)

model_GBoost = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.03,
                                   max_depth=3, max_features=0.4,
                                   min_samples_leaf=20, min_samples_split=10, 
                                   loss='huber', random_state = seed)

model_xgb = XGBRegressor(colsample_bytree=0.35, gamma=0.027, 
                             learning_rate=0.03, max_depth=4, 
                             min_child_weight=1.7817, n_estimators=3000,
                             reg_alpha=0.43, reg_lambda=0.88,
                             subsample=0.5213, silent=1,
                             random_state = seed)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=10,
                              learning_rate=0.03, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



#Linear Models
linear_models = []
linear_models.append(("lasso", model_lasso))
linear_models.append(("ridge", model_ridge))
linear_models.append(("svr", model_svr))
linear_models.append(("ENet", model_ENet))
linear_models.append(("KRR", model_KRR))
linear_models.append(("byr", model_byr))
linear_models.append(("lsvr", model_lsvr))
linear_models.append(("lasso_lars", model_lasso_lars))

tree_models = []
tree_models.append(("rforest", model_rforest))
tree_models.append(("GBoost", model_GBoost))
tree_models.append(("xgb", model_xgb))
tree_models.append(("lgb", model_lgb))

cross_val_table = get_validation_scores(linear_models, X_train, y_train, X_test, y_test)
print(cross_val_table)

linear_cross_val_table = get_validation_scores(linear_models, train_X_reduced, train_y)
print(linear_cross_val_table)

tree_cross_val_table = get_validation_scores(tree_models, train_X_reduced, train_y)
print(tree_cross_val_table)

averaged_models = em.AveragingModels(models = [model_lgb, model_KRR, model_ridge, model_lsvr])
stacked_averaged_models = em.StackingAveragedModels(base_models = [model_KRR, model_lsvr, model_lgb], meta_model = model_ridge)
averaged_plus = em.AveragingModels(models = [averaged_models, model_GBoost, model_xgb], weights = [0.7, 0.2, 0.1])
averaged_plus_plus = em.AveragingModels(models = [stacked_averaged_models, model_GBoost, model_xgb], weights = [0.7, 0.2, 0.1])

avg_full = em.AveragingModels(models = [em.AveragingModels(models = [model_KRR, model_ridge, model_lsvr]), 
                                        em.AveragingModels(models = [model_lgb, model_GBoost, model_xgb])])

ensemble_models = []
ensemble_models.append(("averaged", averaged_models))
ensemble_models.append(("stacked", stacked_averaged_models))
ensemble_models.append(("averaged_plus", averaged_plus))
ensemble_models.append(("averaged_plus_plus", averaged_plus_plus))
ensemble_models.append(("averaged_full", avg_full))

cross_val_table_ensemble = get_validation_scores(ensemble_models, X_train, y_train, X_test, y_test)
print(cross_val_table_ensemble)

cross_val_table_ensemble = get_validation_scores(ensemble_models, train_X_reduced, train_y)
print(cross_val_table_ensemble)

make_submission(averaged_plus, train_X_reduced, train_y, test_X_reduced, filename = 'submission_avg_plus.csv')
make_submission(averaged_models, train_X_reduced, train_y, test_X_reduced, filename = 'submission_avg.csv')
make_submission(averaged_plus_plus, train_X_reduced, train_y, test_X_reduced, filename = 'submission_avg_plus_plus.csv')

make_submission(avg_full, train_X_reduced, train_y, test_X_reduced, filename = 'submission_avg_full.csv')







import featuretools as ft
