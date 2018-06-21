from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler
import preprocessing as pp
import numpy as np
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge

def get_validation_scores(models, X_train, y_train, folds):
    scores_val_mean = []
    scores_val_std = []
    scores_val = []
    names = []
    for name, model in models:
        names.append(name)
        val_scores = np.sqrt(pp.score_model(model, X_train, y_train, n_folds = folds))
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


log_transformer = FunctionTransformer(np.log1p)

train, test = pp.read_train_test(train_file = 'train.csv', test_file = 'test.csv')

ids = list(test.ID)

train.drop(['ID'], axis=1, inplace = True)
test.drop(['ID'], axis=1, inplace = True)

train_y = (np.log1p(train.target)).values

train.drop(['target'], axis=1, inplace = True)

scaler = RobustScaler()

threshold = .98 * (1 - .98)
variance = VarianceThreshold(threshold)

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

percentile = SelectPercentile(mutual_info_regression, percentile=85)

from_model = SelectFromModel(ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0))

feature_selection = []
feature_selection.append(('percentile', percentile))
feature_selection.append(('from_model', from_model))
feature_selection_union = FeatureUnion(feature_selection)

estimators = []
estimators.append(('low_variance', variance))
#estimators.append(('scaler', scaler))
#estimators.append(('anova', anova_filter))
estimators.append(('log_transform', log_transformer))
#estimators.append(('percentile', percentile))
#estimators.append(('from_model', from_model))
#estimators.append(('feature_selection', feature_selection_union))

pipe = Pipeline(estimators)
pipe.fit(train, train_y)

train_X_reduced = pipe.transform(train)
test_X_reduced = pipe.transform(test)


estimators.append(('rf', model_rforest))



model_lgb.fit(train_X_reduced, train_y)

prediction = model_lgb.predict(test_X_reduced)

tree_models = []
tree_models.append(("lgb", model_lgb))
#tree_models.append(("rf", model_rforest))
#tree_models.append(("xgb", model_xgb))
tree_models.append(("byr", model_byr))

cross_val_table = get_validation_scores(tree_models, train_X_reduced, train_y, 5)
print(cross_val_table)


make_submission(model_lgb, train_X_reduced, train_y, test_X_reduced, ids, filename = 'submission.csv')