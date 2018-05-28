import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, Ridge, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR, LinearSVC
import numpy as np
from scipy.stats import skew
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
#from sklearn.decomposition import PCA, KernelPCA

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights = "same"):
        self.models = models
        self.weights = weights
        
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
        if self.weights == "same":
            return np.mean(predictions, axis=1)
        else:
            for i in range(len(self.models_)):
                predictions[:, i] *= self.weights[i]
            return np.sum(predictions, axis=1)

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

class HousePricesGrid():
    def __init__(self, model, hyperparameters, n_folds = 3):
        self.model = model
        self.hyperparameter = hyperparameters
        self.folds = n_folds
    
    def grid_get(self, X, y):
        grid_search = GridSearchCV(self.model, self.hyperparameter,
                                   cv=self.folds, 
                                   scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])

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

#def log_transform(dataset, feature):
#    dataset[feature] = np.log1p(dataset[feature].values)

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

def log_transform(dataset, cols, threshold =0.75):
    skewness = dataset[cols].apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > threshold]
    print(str(skewness.shape[0]) + " skewed numerical features to log transform")
    skewed_features = skewness.index
    dataset[skewed_features] = np.log1p(dataset[skewed_features])

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

ds["OverallQual-s2"] = ds["OverallQual"] ** 2
ds["OverallQual-s3"] = ds["OverallQual"] ** 3
ds["OverallQual-Sq"] = np.sqrt(ds["OverallQual"])
ds["AllSF-2"] = ds["AllSF"] ** 2
ds["AllSF-3"] = ds["AllSF"] ** 3
ds["AllSF-Sq"] = np.sqrt(ds["AllSF"])
ds["AllFlrsSF-2"] = ds["AllFlrsSF"] ** 2
ds["AllFlrsSF-3"] = ds["AllFlrsSF"] ** 3
ds["AllFlrsSF-Sq"] = np.sqrt(ds["AllFlrsSF"])
ds["GrLivArea-2"] = ds["GrLivArea"] ** 2
ds["GrLivArea-3"] = ds["GrLivArea"] ** 3
ds["GrLivArea-Sq"] = np.sqrt(ds["GrLivArea"])
ds["SimplOverallQual-s2"] = ds["SimplOverallQual"] ** 2
ds["SimplOverallQual-s3"] = ds["SimplOverallQual"] ** 3
ds["SimplOverallQual-Sq"] = np.sqrt(ds["SimplOverallQual"])
ds["ExterQual-2"] = ds["ExterQual"] ** 2
ds["ExterQual-3"] = ds["ExterQual"] ** 3
ds["ExterQual-Sq"] = np.sqrt(ds["ExterQual"])
ds["GarageCars-2"] = ds["GarageCars"] ** 2
ds["GarageCars-3"] = ds["GarageCars"] ** 3
ds["GarageCars-Sq"] = np.sqrt(ds["GarageCars"])
ds["TotalBath-2"] = ds["TotalBath"] ** 2
ds["TotalBath-3"] = ds["TotalBath"] ** 3
ds["TotalBath-Sq"] = np.sqrt(ds["TotalBath"])
ds["KitchenQual-2"] = ds["KitchenQual"] ** 2
ds["KitchenQual-3"] = ds["KitchenQual"] ** 3
ds["KitchenQual-Sq"] = np.sqrt(ds["KitchenQual"])
ds["GarageScore-2"] = ds["GarageScore"] ** 2
ds["GarageScore-3"] = ds["GarageScore"] ** 3
ds["GarageScore-Sq"] = np.sqrt(ds["GarageScore"])

num_columns, cat_columns = get_predictors(ds)

print("Numerical features : " + str(len(num_columns)))
print("Categorical features : " + str(len(cat_columns)))

predictors = num_columns + cat_columns

ds = ds[predictors + ['Id','dataset']]
encoded_ds = hot_encode(ds)

log_transform(encoded_ds, num_columns, 0.5)

train_y = np.log1p(train.SalePrice)
train_X = (encoded_ds.loc[encoded_ds.dataset == "train"]).drop(['dataset'], axis=1)
test_X = (encoded_ds.loc[encoded_ds.dataset == "test"]).drop(['dataset'], axis=1)

scaler = RobustScaler()
train_X = scaler.fit(train_X).transform(train_X)
test_X = scaler.transform(test_X)

#########################################################################################################
hyperparameters_ridge = {'alpha': list(np.arange(9.4, 9.51, 0.01))}
hyperparameters_lasso = {'alpha': list(np.arange(0.0004, 0.00061, 0.00005)), 'max_iter':[10000]}
hyperparameters_svr = {'C':[11, 13, 15], 'kernel':["rbf"], 
                       "gamma":[0.0003, 0.0004], 
                       "epsilon":[0.008, 0.009]}
hyperparameters_krr = {'alpha':[0.5, 0.6], 'kernel':["polynomial"], 'degree':[2], 'coef0':[2, 2.5]}
hyperparameters_ENet = {'alpha':[0.0008, 0.004, 0.0005],'l1_ratio':[0.08, 0.1, 0.9],'max_iter':[10000]}

hpg = HousePricesGrid(ElasticNet(), hyperparameters = hyperparameters_ENet)
hpg.grid_get(train_X, train_y)


model_ridge = Ridge(alpha=9.48)
model_lasso = Lasso(alpha =0.0005, random_state=1)
model_svr = SVR(C = 15, epsilon = 0.009, gamma = 0.0004, kernel = 'rbf')
model_KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
model_ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)


model_lsvr = LinearSVR()

model_sgd = SGDRegressor()

model_byr = BayesianRidge()

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
model_rforest = RandomForestRegressor(50)

models = [model_ridge, model_xgb, model_lasso, model_ENet, model_KRR, 
          model_GBoost, model_lgb, model_rforest, model_svr, model_lsvr,
          model_sgd, model_byr, model_extra]
names = ["ridge", "xgb", "lasso", "ENet", "KRR", "GBoost",
         "lgb", "rforest", "svr", "lsvr", "sgd", "byr", "extra"]

for name, model in zip(names, models):
    score = np.sqrt(score_model(model, train_X, train_y))
    print("{} score: {:.4f} ({:.4f})\n".format(name, score.mean(), score.std()))

for model in models:
    model.fit(train_X, train_y)

averaged_models = AveragingModels(models = (model_lasso, model_ENet, model_svr, model_byr, model_ridge, model_GBoost, model_lgb), 
                                  weights = [0.25, 0.25, 0.2, 0.1, 0.1, 0.05, 0.05])

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

predicted_prices = predicted_prices_stacked_averaged*0.7 + predicted_prices_xgboost*0.15 + predicted_prices_lgb*0.15

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_ensemble.csv', index=False)




