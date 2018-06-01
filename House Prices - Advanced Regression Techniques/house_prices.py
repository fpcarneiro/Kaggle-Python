import preprocessing as pp
import ensemble as em
import cv_lab as cvl
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.svm import SVR, LinearSVR, LinearSVC
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

train, test = pp.read_train_test()

train = pp.drop_outliers(train)

all_data = pp.concat_train_test(train.drop(['SalePrice'], axis=1), test)

#ds = ds.drop(['Utilities'], axis=1)
#ds = ds.drop(high_occurance_missing(ds, 0.8), axis=1)

all_data = pp.convert_numeric2category(all_data)
was_missing_columns = pp.handle_missing(all_data)

all_data = pp.encode(all_data)
shrunk_columns = pp.shrink_scales(all_data)
engineered_columns = pp.add_engineered_features(all_data)
simplified_columns = pp.simplify_features(all_data)
polinomial_columns = pp.polinomial_features(all_data)

num_columns, cat_columns = pp.get_feature_groups(all_data)

print("Numerical features : " + str(len(num_columns)))
print("Categorical features : " + str(len(cat_columns)))

engineered_columns = list(set(engineered_columns) - set(cat_columns))

all_data_encoded = pp.hot_encode(all_data)

pp.log_transform(all_data_encoded, list(set(num_columns)-set(was_missing_columns)), 0.5)

train_y = np.log1p(train.SalePrice)
train_X = (all_data_encoded.loc[all_data_encoded.dataset == "train"]).drop(['dataset', 'Id'], axis=1)
test_X = (all_data_encoded.loc[all_data_encoded.dataset == "test"]).drop(['dataset', 'Id'], axis=1)

#########################################################################################################
all_predictors = train_X.columns
predictors = list(set(train_X.columns) - set(was_missing_columns))

scaler = RobustScaler()
train_X = scaler.fit(train_X[predictors]).transform(train_X[predictors])
test_X = scaler.transform(test_X[predictors])
train_y = train_y.as_matrix()

sfm = SelectFromModel(Lasso(alpha = 0.000507))
#sfm = SelectFromModel(BayesianRidge())

sfm.fit(train_X, train_y)
train_X_reduced = sfm.transform(train_X)
test_X_reduced = sfm.transform(test_X)
#########################################################################################################

model_lasso = Lasso(alpha = 0.000507, random_state = 1)
model_ridge = Ridge(alpha=10.0)
#model_svr = SVR(C = 1.466, epsilon = 0.0322, gamma = 0.0015, kernel = 'rbf')
model_svr = SVR(C = 15, epsilon = 0.009, gamma = 0.0004, kernel = 'rbf')
model_ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3, max_iter = 10000)
model_KRR = KernelRidge(alpha=0.5, kernel='polynomial', degree=2, coef0=2.5)
model_byr = BayesianRidge()
model_rforest = RandomForestRegressor(n_estimators = 210)

model_lsvr = LinearSVR()
model_sgd = SGDRegressor()
model_extra = ExtraTreesRegressor()

model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=4, 
                             min_child_weight=1.7817, n_estimators=2500,
                             reg_alpha=0.4640, reg_lambda=0.88,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

models = []
models.append(model_lasso)
models.append(model_ridge)
models.append(model_svr)
models.append(model_ENet)
models.append(model_KRR)
models.append(model_byr)
models.append(model_rforest)
models.append(model_xgb)
models.append(model_GBoost)
models.append(model_lgb)

names = ["lasso", "ridge", "svr", "ENet", "KRR", "byr", "rforest", "xgb", "GBoost", "lgb"]
#names = ["ridge", "xgb", "lasso", "ENet", "KRR", "GBoost",
#         "lgb", "rforest", "svr", "lsvr", "sgd", "byr", "extra"]
   
scores = [(np.sqrt(pp.score_model(model, train_X_reduced, train_y))).mean() for name, model in zip(names, models)]
tab = pd.DataFrame({ "Model" : names, "Score" : scores })
tab = tab.sort_values(by=['Score'], ascending = True)
print(tab)

for model in models:
    model.fit(train_X, train_y)
    
averaged_models = em.AveragingModels(models = (model_lasso, model_ENet, model_svr, model_KRR, 
                                            model_ridge, model_byr, model_xgb, model_GBoost) )

score_avg = np.sqrt(pp.score_model(averaged_models, train_X_reduced, train_y))
print(" Averaged base models score: {:.6f} ({:.6f})\n".format(score_avg.mean(), score_avg.std()))

averaged_models.fit(train_X_reduced, train_y)
predicted_prices_averaged = np.expm1(averaged_models.predict(test_X_reduced))
print(predicted_prices_averaged)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices_averaged})
my_submission.to_csv('submission_avg.csv', index=False)

stacked_averaged_models = em.StackingAveragedModels(base_models = (model_ENet, model_svr, model_KRR, model_lasso,
                                                                model_ridge),
                                                 meta_model = model_byr)

score_stacked_averaged = np.sqrt(pp.score_model(stacked_averaged_models, train_X_reduced, train_y))
print(" Stacked Averaged base models score: {:.6f} ({:.6f})\n".format(score_stacked_averaged.mean(), 
      score_stacked_averaged.std()))

stacked_averaged_models.fit(train_X_reduced, train_y)
predicted_prices_stacked_averaged = np.expm1(stacked_averaged_models.predict(test_X_reduced))
print(predicted_prices_stacked_averaged)

model_xgb.fit(train_X_reduced, train_y)
predicted_prices_xgboost = np.expm1(model_xgb.predict(test_X_reduced))
print(predicted_prices_xgboost)

model_lgb.fit(train_X_reduced, train_y)
predicted_prices_lgb = np.expm1(model_lgb.predict(test_X_reduced))
print(predicted_prices_lgb)

model_GBoost.fit(train_X_reduced, train_y)
predicted_prices_GBoost = np.expm1(model_GBoost.predict(test_X_reduced))
print(predicted_prices_GBoost)

predicted_prices = predicted_prices_stacked_averaged*0.7 + predicted_prices_xgboost*0.2 + predicted_prices_lgb*0.1

predicted_prices = (predicted_prices_stacked_averaged*0.6 + predicted_prices_averaged*0.4)*0.7 + predicted_prices_xgboost*0.2 + predicted_prices_lgb*0.1

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_ensemble.csv', index=False)