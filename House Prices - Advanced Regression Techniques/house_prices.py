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
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, Ridge, SGDRegressor, LassoLars
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

train_X, train_y, test_X = pp.get_processed_datasets()

#########################################################################################################
all_predictors = train_X.columns
predictors = list(set(train_X.columns) - set(was_missing_columns))

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
plt.style.use('ggplot')
lasso = Lasso(alpha=0.000507)
lasso.fit(train_X, train_y)
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=train_X.columns)
FI_lasso.sort_values("Feature Importance",ascending=False)
FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()

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
                             min_child_weight=1.7817, n_estimators=3000,
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

model_lasso_lars = LassoLars(alpha = 0.000507)

#Linear Models
models = []
models.append(("lasso", model_lasso))
models.append(("ridge", model_ridge))
models.append(("svr", model_svr))
models.append(("ENet", model_ENet))
models.append(("KRR", model_KRR))
models.append(("byr", model_byr))
models.append(("rforest", model_rforest))
models.append(("xgb", model_xgb))
models.append(("GBoost", model_GBoost))
models.append(("lgb", model_lgb))
models.append(("lasso_lars", model_lasso_lars))
models.append(("lsvr", model_lsvr))
#models.append(("sgd", model_sgd))
#models.append(("extra", model_extra))

scores = []
names = []
for name, model in models:
    names.append(name)
    scores.append(np.sqrt(pp.score_model(model, train_X_reduced, train_y)).mean())
tab = pd.DataFrame({ "Model" : names, "Score" : scores })
tab = tab.sort_values(by=['Score'], ascending = True)
print(tab)

for name, model in models:
    model.fit(train_X_reduced, train_y)
    
averaged_models = em.AveragingModels(models = [model_byr, model_ENet, model_KRR])

score_avg = np.sqrt(pp.score_model(averaged_models, train_X_reduced, train_y))
print(" Averaged base models score: {:.6f} ({:.6f})\n".format(score_avg.mean(), score_avg.std()))

averaged_models.fit(train_X_reduced, train_y)
predicted_prices_averaged = np.expm1(averaged_models.predict(test_X_reduced))
print(predicted_prices_averaged)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices_averaged})
my_submission.to_csv('submission_avg.csv', index=False)

stacked_averaged_models = em.StackingAveragedModels(base_models = [model_byr, model_ENet, model_lasso],
                                                 meta_model = model_KRR)

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

predicted_prices = predicted_prices_stacked_averaged*0.7 + predicted_prices_xgboost*0.2 + predicted_prices_lgb*0.1

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_ensemble.csv', index=False)




