import preprocessing as pp
import ensemble as em
import transformers as tf

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.feature_selection import SelectPercentile, SelectFromModel, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, Lasso, Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from xgboost import XGBRegressor

train, test = pp.read_train_test(train_file = 'train.csv', test_file = 'test.csv')

ids = list(test.ID)

train_X = train.drop(['ID','target'], axis=1)
train_y = (np.log1p(train.target)).values

test_X = test.drop(['ID'], axis=1)

scaler = RobustScaler()

threshold = .98 * (1 - .98)
variance = VarianceThreshold(threshold)

int64_columns = list(train_X.select_dtypes(include=['int64']).columns)
float64_columns = list(train_X.select_dtypes(include=['float64']).columns)

binarizer = tf.BinarizerTransformer(columns=int64_columns)

anova_filter = SelectKBest(f_regression, k=4000)

model_rforest = RandomForestRegressor(n_estimators = 50, 
                                      max_features = 0.7,
                                      random_state=2018,
                                      max_depth=20,
                                      min_samples_leaf = 4,
                                      min_samples_split = 10)

model_lgb = lgb.LGBMRegressor(objective='regression',
                              metric="rmse",
                              n_estimators = 500,
                              num_leaves = 30,
                              learning_rate = 0.01,
                              bagging_fraction = 0.7,
                              feature_fraction = 0.7,
                              bagging_frequency = 5,
                              bagging_seed = 2018,
                              verbosity = -1)


model_xgb = XGBRegressor(n_estimators = 100, 
                         colsample_bytree = 0.7,
                         colsample_bylevel = 0.7,
                         learning_rate=0.1)

model_byr = BayesianRidge()

percentile = SelectPercentile(mutual_info_regression, percentile=35)

from_model_lasso = SelectFromModel(Lasso())
from_model_extra_tree = SelectFromModel(ExtraTreesRegressor(n_estimators=200, max_depth=20, 
                                                            max_features=0.5, n_jobs=-1, random_state=0))
from_model_lgb = SelectFromModel(model_lgb)

log_transformer = FunctionTransformer(np.log1p)

feature_selection = []
#feature_selection.append(('percentile', percentile))
#feature_selection.append(('from_model_lasso', from_model_lasso))
#feature_selection.append(('from_model_extra_tree', from_model_extra_tree))
feature_selection.append(('from_model_lgb', from_model_lgb))
feature_selection_union = FeatureUnion(feature_selection)

estimators = []
#estimators.append(('binarizer', binarizer))
estimators.append(('low_variance', variance))
#estimators.append(('scaler', scaler))
#estimators.append(('anova', anova_filter))
#estimators.append(('log_transform', log_transformer))
#estimators.append(('percentile', percentile))
#estimators.append(('from_model', from_model))
#estimators.append(('feature_selection', feature_selection_union))

pipe = Pipeline(estimators)
pipe.fit(train_X, train_y)

train_X_reduced = pipe.transform(train_X)
test_X_reduced = pipe.transform(test_X)

print(train_X_reduced.shape)
print(test_X_reduced.shape)

train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(train_X_reduced, train_y, test_size=0.1)

train_set_X.shape

tree_models = []
tree_models.append(("lgb", model_lgb))
#tree_models.append(("rf", model_rforest))
#tree_models.append(("xgb", model_xgb))

cross_val_table = pp.get_validation_scores(tree_models, train_set_X, train_set_y, 3)
print(cross_val_table)


averaged_models = em.AveragingModels(models = [model_lgb, model_rforest])

ensemble_models = []
ensemble_models.append(("averaged", averaged_models))

cross_val_table_avg = pp.get_validation_scores(ensemble_models, train_X_reduced, train_y, 5)
print(cross_val_table_avg)

pp.make_submission(model_lgb, train_X_reduced, train_y, test_X_reduced, ids, filename = 'submission.csv')





(objective='regression',
                              metric="rmse",
                              n_estimators = 500,
                              num_leaves = 30,
                              learning_rate = 0.01,
                              bagging_fraction = 0.7,
                              feature_fraction = 0.7,
                              bagging_frequency = 5,
                              bagging_seed = 2018,
                              verbosity = -1)


lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    "learning_rate": 0.01,
    "num_leaves": 180,
    "feature_fraction": 0.50,
    "bagging_fraction": 0.50,
    'bagging_freq': 4,
    "max_depth": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    #"min_split_gain":0.2,
    "min_child_weight":10,
    'zero_as_missing':True
                }

model_lgb = lgb.LGBMRegressor(params = lgbm_params)

lgb_train = lgb.Dataset(train_set_X, train_set_y, feature_name = "auto")


lgb_cv = lgb.cv(
    params = lgbm_params,
    train_set = lgb_train,
    num_boost_round=2000,
    stratified=False,
    nfold = 5,
    verbose_eval=50,
    seed = 23,
    early_stopping_rounds=75)

results = pd.DataFrame(columns = ["Rounds","Score","STDV", "LB", "Parameters"])
optimal_rounds = np.argmin(lgb_cv['rmse-mean'])
best_cv_score = min(lgb_cv['rmse-mean'])

print("\nOptimal Round: {}\nOptimal Score: {} + {}".format(
    optimal_rounds,best_cv_score,lgb_cv['rmse-stdv'][optimal_rounds]))

results = results.append({"Rounds": optimal_rounds,
                          "Score": best_cv_score,
                          "STDV": lgb_cv['rmse-stdv'][optimal_rounds],
                          "LB": None,
                          "Parameters": lgbm_params}, ignore_index=True)