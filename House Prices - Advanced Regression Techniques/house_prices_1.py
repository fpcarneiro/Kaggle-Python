import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from scipy.stats import skew
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=3):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred        
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

DATADIR = "input/"

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

def score_dataset(train_X, test_X, train_y, test_y):
    forest_model = RandomForestRegressor()
    forest_model.fit(train_X, train_y)
    preds = forest_model.predict(test_X)
    return(mean_absolute_error(test_y, preds))

def read_train_test(train_file = 'train.csv', test_file = 'test.csv'):
    train = pd.read_csv(DATADIR + train_file)
    test = pd.read_csv(DATADIR + test_file)
    return train, test

def concat_train_test(train, test, ignore_index=False):
    dataset = train.append(test, ignore_index=ignore_index)
    dataset["dataset"] = "train"
    dataset.loc[dataset.Id.isin(test.Id), "dataset"] = "test"
    return dataset

def convert_numeric2category(dataset):
    # Some numerical features are actually really categories
    mydata = dataset.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })
    return(mydata)

def high_occurance_missing(dataset, threshold):    
    return([c for c in list(dataset.columns) if ( dataset[c].isnull().sum() / len(dataset) ) >= threshold])

def get_predictors(dataset, drop_list = ['dataset', 'Id']):
    mydata = dataset.drop(drop_list, axis=1)
    num_columns = list(mydata.select_dtypes(exclude=['object']).columns)
    cat_columns = list(mydata.select_dtypes(include=['object']).columns)
    return (num_columns, cat_columns)

def hot_encode(dataset, drop_list = ['dataset', 'Id']):
    encoded = pd.get_dummies(dataset.drop(drop_list, axis=1))
    #return (pd.merge(encoded, dataset[['Id'] + drop_list], how='inner', on=['Id']))
    return (pd.concat([ dataset[drop_list] , encoded], axis=1))

def log_transform(dataset, feature):
    dataset[feature] = np.log1p(dataset[feature].values)

def quadratic(dataset, feature):
    dataset[feature+'2'] = dataset[feature]**2

def check_missing(dataset):
    all_data_na = (dataset.isnull().sum() / len(dataset)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    return(missing_data)

def handle_missing(dataset):
    cols_mode = ['MSZoning', 'SaleType', 'Electrical', 'Exterior1st', 'Exterior2nd']
    no_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtQual",
               "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType"]
    zero_cols = ["GarageYrBlt", "LotFrontage", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath",
                 "BsmtHalfBath", "MasVnrArea"]
    
    missing_dict = dict(zip(zero_cols,[0] * len(zero_cols)))
    missing_dict.update(dict(zip(no_cols,["No"] * len(no_cols))))
    missing_dict["Functional"] = "Typ"
    missing_dict["Utilities"] = "AllPub"
    missing_dict["KitchenQual"] = "TA"
    
    for (k, v) in missing_dict.items():
        dataset.loc[:, k] = dataset.loc[:, k].fillna(v)

    for col in cols_mode:
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

def encode(dataset):
    quality_scale = {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}
    basement_scale = {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}
    access_scale = {"No" : 0, "Grvl" : 1, "Pave" : 2}
    
    replace_table = {"Alley" : access_scale,
                       "BsmtCond" : quality_scale,
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : basement_scale,
                       "BsmtFinType2" : basement_scale,
                       "BsmtQual" : quality_scale,
                       "ExterCond" : quality_scale,
                       "ExterQual" : quality_scale,
                       "FireplaceQu" : quality_scale,
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : quality_scale,
                       "GarageQual" : quality_scale,
                       "HeatingQC" : quality_scale,
                       "KitchenQual" : quality_scale,
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : quality_scale,
                       "Street" : access_scale,
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
# Encode some categorical features as ordered numbers when there is information in the order
    mydata = dataset.replace(to_replace = replace_table)
    return (mydata)

def simplify_features1(dataset):
    # Create new features
    overall_scale = {1 : 1, 2 : 1, 3 : 1, # bad
                     4 : 2, 5 : 2, 6 : 2, # average
                     7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                     }
    
    quality_scale = {1 : 1, # bad
                     2 : 1, 3 : 1, # average
                     4 : 2, 5 : 2, 6 : 2 # good
                     }
    
    simpl_dict = {}
    simpl_dict["OverallQual"] = overall_scale
    simpl_dict["OverallCond"] = overall_scale
    simpl_dict["PoolQC"] = quality_scale
    simpl_dict["GarageCond"] = quality_scale
    simpl_dict["GarageQual"] = quality_scale
    simpl_dict["FireplaceQu"] = quality_scale
    simpl_dict["Functional"] = {1 : 1, 2 : 1, # bad
                            3 : 2, 4 : 2, # major
                            5 : 3, 6 : 3, 7 : 3, # minor
                            8 : 4 # typical
                            }
    simpl_dict["KitchenQual"] = quality_scale
    simpl_dict["HeatingQC"] = quality_scale
    simpl_dict["BsmtFinType1"] = quality_scale
    simpl_dict["BsmtFinType2"] = quality_scale
    simpl_dict["BsmtCond"] = quality_scale
    simpl_dict["BsmtQual"] = quality_scale
    simpl_dict["ExterCond"] = quality_scale
    simpl_dict["ExterQual"] = quality_scale
     
    for (k,v) in simpl_dict.items():
        ds["Simpl"+ k] = ds[k].replace(v)

def simplify_features2(dataset):
## 2* Combinations of existing features
## Overall quality of the house
    dataset["OverallGrade"] = dataset["OverallQual"] * dataset["OverallCond"]
## Overall quality of the garage
    dataset["GarageGrade"] = dataset["GarageQual"] * dataset["GarageCond"]
#    # Overall quality of the exterior
    dataset["ExterGrade"] = dataset["ExterQual"] * dataset["ExterCond"]
#    # Overall kitchen score
    dataset["KitchenScore"] = dataset["KitchenAbvGr"] * dataset["KitchenQual"]
#    # Overall fireplace score
    dataset["FireplaceScore"] = dataset["Fireplaces"] * dataset["FireplaceQu"]
#    # Overall garage score
    dataset["GarageScore"] = dataset["GarageArea"] * dataset["GarageQual"]
#    # Overall pool score
    dataset["PoolScore"] = dataset["PoolArea"] * dataset["PoolQC"]
#    # Simplified overall quality of the house
    dataset["SimplOverallGrade"] = dataset["SimplOverallQual"] * dataset["SimplOverallCond"]
#    # Simplified overall quality of the exterior
    dataset["SimplExterGrade"] = dataset["SimplExterQual"] * dataset["SimplExterCond"]
#    # Simplified overall pool score
    dataset["SimplPoolScore"] = dataset["PoolArea"] * dataset["SimplPoolQC"]
#    # Simplified overall garage score
    dataset["SimplGarageScore"] = dataset["GarageArea"] * dataset["SimplGarageQual"]
#    # Simplified overall fireplace score
    dataset["SimplFireplaceScore"] = dataset["Fireplaces"] * dataset["SimplFireplaceQu"]
#    # Simplified overall kitchen score
    dataset["SimplKitchenScore"] = dataset["KitchenAbvGr"] * dataset["SimplKitchenQual"]
#    # Total number of bathrooms
    dataset["TotalBath"] = dataset["BsmtFullBath"] + (0.5 * dataset["BsmtHalfBath"]) + \
    dataset["FullBath"] + (0.5 * dataset["HalfBath"])
#    # Total SF for house (incl. basement)
    dataset["AllSF"] = dataset["GrLivArea"] + dataset["TotalBsmtSF"]
#    # Total SF for 1st + 2nd floors
    dataset["AllFlrsSF"] = dataset["1stFlrSF"] + dataset["2ndFlrSF"]
#    # Total SF for porch
    dataset["AllPorchSF"] = dataset["OpenPorchSF"] + dataset["EnclosedPorch"] + \
    dataset["3SsnPorch"] + dataset["ScreenPorch"]
## Has masonry veneer or not
    dataset["HasMasVnr"] = dataset.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                               "Stone" : 1, "None" : 0})
#    # House completed before sale or not
    dataset["BoughtOffPlan"] = dataset.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})
    #dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
    
#obj_df["num_cylinders"].value_counts()
    
def score_model(estimator, X, y, n_folds = 3, scoring_func="neg_mean_squared_error"):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
    score = -cross_val_score(estimator, X, y, scoring=scoring_func, cv = kf)
    return(score)

train, test = read_train_test()

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
#train = train.drop(['Utilities'], axis=1)

ds = concat_train_test(train.drop(['SalePrice'], axis=1), test)

#ds = ds.drop(high_occurance_missing(ds, 0.8), axis=1)
ds = convert_numeric2category(ds)
handle_missing(ds)

ds = encode(ds)
simplify_features1(ds)
simplify_features2(ds)

num_columns, cat_columns = get_predictors(ds)

print("Numerical features : " + str(len(num_columns)))
print("Categorical features : " + str(len(cat_columns)))

predictors = num_columns + cat_columns

ds = ds[predictors + ['Id','dataset']]
encoded_ds = hot_encode(ds)

skewness = encoded_ds[num_columns].apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
encoded_ds[skewed_features] = np.log1p(encoded_ds[skewed_features])

train_y = np.log1p(train.SalePrice)
train_X = (encoded_ds.loc[encoded_ds.dataset == "train"]).drop(['dataset'], axis=1)
test_X = (encoded_ds.loc[encoded_ds.dataset == "test"]).drop(['dataset'], axis=1)

#########################################################################################################

model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=4, 
                             min_child_weight=1.7817, n_estimators=2500,
                             reg_alpha=0.4640, reg_lambda=0.88,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
model_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
model_KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
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
model_rforest = RandomForestRegressor(50)

score_xgb = np.sqrt(score_model(model_xgb, train_X, train_y))
score_lasso = np.sqrt(score_model(model_lasso, train_X, train_y))
score_ENet = np.sqrt(score_model(model_ENet, train_X, train_y))
score_KRR = np.sqrt(score_model(model_KRR, train_X, train_y))
score_GBoost = np.sqrt(score_model(model_GBoost, train_X, train_y))
score_model_lgb = np.sqrt(score_model(model_lgb, train_X, train_y))
score_forest = np.sqrt(score_model(model_rforest, train_X, train_y))

print("\nXGBoost score: {:.4f} ({:.4f})\n".format(score_xgb.mean(), score_xgb.std()))
print("\nLasso score: {:.4f} ({:.4f})\n".format(score_lasso.mean(), score_lasso.std()))
print("\nEnet score: {:.4f} ({:.4f})\n".format(score_ENet.mean(), score_ENet.std()))
print("\nKRR score: {:.4f} ({:.4f})\n".format(score_KRR.mean(), score_KRR.std()))
print("\nGBoost score: {:.4f} ({:.4f})\n".format(score_GBoost.mean(), score_GBoost.std()))
print("\nmodel_lgb score: {:.4f} ({:.4f})\n".format(score_model_lgb.mean(), score_model_lgb.std()))
print("\nRandom Forest score: {:.4f} ({:.4f})\n".format(score_forest.mean(), score_model_lgb.std()))

model_xgb.fit(train_X, train_y)
model_lasso.fit(train_X, train_y)
model_ENet.fit(train_X, train_y)
model_KRR.fit(train_X, train_y)
model_GBoost.fit(train_X, train_y)
model_lgb.fit(train_X, train_y)
model_rforest.fit(train_X, train_y)


averaged_models = AveragingModels(models = (model_ENet, model_GBoost, model_xgb, model_lgb, model_lasso))

score_avg = np.sqrt(score_model(averaged_models, train_X, train_y))
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score_avg.mean(), score_avg.std()))

averaged_models.fit(train_X, train_y)
predicted_prices = np.expm1(averaged_models.predict(test_X))
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_avg.csv', index=False)


stacked_averaged_models = StackingAveragedModels(base_models = (model_ENet, model_GBoost),
                                                 meta_model = model_lasso)

score_stacked_averaged = np.sqrt(score_model(stacked_averaged_models, train_X, train_y))
print(" Stacked Averaged base models score: {:.4f} ({:.4f})\n".format(score_stacked_averaged.mean(), score_stacked_averaged.std()))

stacked_averaged_models.fit(train_X, train_y)
predicted_prices_stacked_averaged = np.expm1(stacked_averaged_models.predict(test_X))
print(predicted_prices_stacked_averaged)

predicted_prices_xgboost = np.expm1(model_xgb.predict(test_X))
print(predicted_prices_xgboost)

predicted_prices_lgb = np.expm1(model_lgb.predict(test_X))
print(predicted_prices_lgb)

predicted_prices = predicted_prices_stacked_averaged*0.6 + predicted_prices_xgboost*0.2 + predicted_prices_lgb*0.2

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_ensemble.csv', index=False)











from sklearn.model_selection import GridSearchCV

hyperparameters = { 'n_estimators': [2500],
                    'max_depth': [4],
                   'learning_rate': [0.05],
                    'reg_lambda': [0.88, 0.885]
                  }
clf = GridSearchCV(model_xgb, hyperparameters, cv=3)
clf.fit(train_X, train_y)
clf.best_params_
clf.refit
score_clf = np.sqrt(score_model(clf, train_X, n_folds = 3, scoring_func="neg_mean_squared_error"))
print(score_clf.mean())