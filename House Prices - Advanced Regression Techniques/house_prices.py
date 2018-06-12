import preprocessing as pp
import transformers as tr
import ensemble as em
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

def get_validation_scores(models, X, y):
    scores = []
    names = []
    for name, model in models:
        names.append(name)
        scores.append(np.sqrt(pp.score_model(model, X, y)).mean())
    tab = pd.DataFrame({ "Model" : names, "Score" : scores })
    tab = tab.sort_values(by=['Score'], ascending = True)
    return(tab)

def make_submission(model, X_train, y_train, X_test, filename = 'submission.csv'):
    model.fit(X_train, y_train)
    predicted = np.expm1(model.predict(X_test))
    print(predicted)
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
                 ('more_features', tr.MoreFeaturesTransformer()),
                 ('encode', tr.EncodeTransformer(prefix = "Shrunk_")),
                 ('feature_engineering', tr.FeatureEngineeringTransformer()),
                 ('simplified_features', tr.SimplifiedFeatureTransformer(prefix = "Shrunk_"))])
    
train = basic_pipeline.fit_transform(train)
test = basic_pipeline.fit_transform(test)

polinomial_transformation = tr.PolinomialFeaturesTransformer(["OverallQual", "AllSF", "AllFlrsSF", "GrLivArea", "Shrunk_OverallQual",
            "ExterQual", "GarageCars", "TotalBath", "KitchenQual", "GarageScore"])
polinomial_transformation.fit(train)
train = polinomial_transformation.transform(train)
test = polinomial_transformation.transform(test)

num_columns_2 = list(train.select_dtypes(exclude=['object']).columns)
cat_columns_2 = list(train.select_dtypes(include=['object']).columns)
                             
second_pipeline = Pipeline([
                 ('have_stuff_features', tr.HaveStuffTransformer()),
                 ('hot_encode', tr.HotEncodeTransformer()),
                 ])

train = second_pipeline.fit_transform(train)
test = second_pipeline.fit_transform(test)

train, test = train.align(test,join='outer', axis=1, fill_value = 0)

log_transformation = tr.LogTransformer(num_columns_2, threshold = 0.5)
log_transformation.fit(train)
train = log_transformation.transform(train)
test = log_transformation.transform(test)

features_variance = fs.list_features_low_variance(train, train_y)

train_X = train[features_variance]
test_X = test[features_variance]

importances = fs.get_feature_importance(Lasso(alpha=0.000507), train_X, train_y)
fs.plot_features_importances(importances, show_importance_zero = False)

features_select_from_model, pipe_select_from_model = fs.remove_features_from_model(estimator = Lasso(alpha=0.000507), 
                                                          scaler = RobustScaler(), X = train_X, y = train_y)
train_X_reduced = pipe_select_from_model.transform(train_X)
test_X_reduced = pipe_select_from_model.transform(test_X)


##################
model_lasso = Lasso(alpha=0.000507, random_state = 1)
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

model_lasso_lars = LassoLars(alpha = 0.17321583392)

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

cross_val_table = get_validation_scores(models, train_X_reduced, train_y)
print(cross_val_table)

averaged_models = em.AveragingModels(models = [model_svr, model_KRR, model_lgb])
stacked_averaged_models = em.StackingAveragedModels(base_models = [model_KRR, model_lgb], meta_model = model_svr)
averaged_plus = em.AveragingModels(models = [averaged_models, model_GBoost, model_xgb], weights = [0.7, 0.2, 0.1])
averaged_plus_plus = em.AveragingModels(models = [stacked_averaged_models, model_GBoost, model_xgb], weights = [0.7, 0.2, 0.1])

ensemble_models = []
ensemble_models.append(("averaged", averaged_models))
ensemble_models.append(("stacked", stacked_averaged_models))
ensemble_models.append(("averaged_plus", averaged_plus))
ensemble_models.append(("stacked_plus_plus", averaged_plus_plus))

cross_val_table_ensemble = get_validation_scores(ensemble_models, train_X_reduced, train_y)
print(cross_val_table_ensemble)

make_submission(averaged_plus, train_X_reduced, train_y, test_X_reduced)