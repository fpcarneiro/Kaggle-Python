import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, RidgeCV
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
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
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
    def __init__(self, model, hyperparameters, n_folds = 5, random = True, random_iter = 10):
        self.model = model
        self.hyperparameter = hyperparameters
        self.folds = n_folds
        self.random = random
        self.random_iter = random_iter
        if self.random:
            self.grid_search = RandomizedSearchCV(self.model, self.hyperparameter,
                                   cv=self.folds, 
                                   scoring = "neg_mean_squared_error",
                                   n_iter = self.random_iter)
        else:
            self.grid_search = GridSearchCV(self.model, self.hyperparameter,
                                   cv=self.folds, 
                                   scoring="neg_mean_squared_error")
    
    def fit(self, X, y):
        self.grid_search.fit(X, y)
        self.grid_search.cv_results_['mean_test_score'] = np.sqrt(-(self.grid_search.cv_results_['mean_test_score']))
        self.grid_search.cv_results_['mean_train_score'] = np.sqrt(-(self.grid_search.cv_results_['mean_train_score']))
        
    def get_best_results(self):
        results = self.grid_search.cv_results_
        print(self.grid_search.best_params_, self.grid_search.best_score_)
        print(pd.DataFrame(results)[['params','mean_test_score','mean_test_score','std_test_score']])
    
    def plot_scores(self):
        results = self.grid_search.cv_results_
        scorer = "score"
        color = 'g'
        # Get the regular numpy array from the MaskedArray
        for parameter, param_range in dict.items(self.hyperparameter):
            
            plt.figure(figsize=(8, 8))
            plt.title("GridSearchCV Evaluating", fontsize=16)
            X_axis = np.array(results['param_' + parameter].data, dtype=float)
            
            ax = plt.axes()
            ax.set_xlim(min(X_axis), max(X_axis))
            ax.set_ylim(0.0, 0.20)
            
            for sample, style in (('train', '--'), ('test', '-')):
                print("")
                print(parameter)
                print(sample)
                sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
                print(len(sample_score_mean))
                sample_score_std = results['std_%s_%s' % (sample, scorer)]
                ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
                ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))
            
            best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
            best_score = results['mean_test_%s' % scorer][best_index]
            
            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
            
            # Annotate the best score for that scorer
            ax.annotate("%0.6f" % best_score, (X_axis[best_index], best_score + 0.005))
            
            plt.xlabel(parameter)
            plt.ylabel("Score")
            plt.grid()
        
            plt.legend(loc="best")
            plt.grid('off')
            plt.show()

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

def get_feature_groups(dataset, drop_list = ['dataset', 'Id']):
    mydata = dataset.drop(drop_list, axis=1)
    num_columns = list(mydata.select_dtypes(exclude=['object']).columns)
    cat_columns = list(mydata.select_dtypes(include=['object']).columns)
    return (num_columns, cat_columns)

def hot_encode(dataset, drop_list = ['dataset', 'Id']):
    encoded = pd.get_dummies(dataset.drop(drop_list, axis=1))
    return (pd.concat([ dataset[drop_list] , encoded], axis=1))

def quadratic(dataset, feature):
    dataset[feature+'2'] = dataset[feature]**2

def check_missing(dataset):
    all_data_na = (dataset.isnull().sum() / len(dataset)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    return(missing_data)

def handle_missing(dataset, add_was_missing_columns = True):
    
    if add_was_missing_columns :
        cols_added = add_columns_was_missing(ds)
    
    cols_mode = ['MSZoning', 'SaleType', 'Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'Utilities', 'KitchenQual']
    no_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu", 
               "GarageQual", "GarageCond", "GarageFinish", "GarageType", "BsmtExposure", 
               "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2"]
    zero_cols = ["MasVnrArea", "BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "GarageArea", "GarageCars"]
    
    missing_dict = dict(zip(zero_cols,[0] * len(zero_cols)))
    missing_dict.update(dict(zip(no_cols,["No"] * len(no_cols))))
    
    for (k, v) in missing_dict.items():
        dataset.loc[:, k] = dataset.loc[:, k].fillna(v)

    for col in cols_mode:
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
    
    dataset.loc[dataset.GarageYrBlt.isnull(),'GarageYrBlt'] = dataset.loc[dataset.GarageYrBlt.isnull(),'YearBuilt']
    
    df_frontage = pd.get_dummies(dataset)
    lf_train = df_frontage.dropna()
    lf_train_y = lf_train.LotFrontage
    lf_train_X = lf_train.drop('LotFrontage',axis=1)
    lr = Ridge()
    lr.fit(lf_train_X, lf_train_y)
    nan_frontage = dataset.LotFrontage.isnull()
    X = df_frontage[nan_frontage].drop('LotFrontage',axis=1)
    y = lr.predict(X)
    dataset.loc[nan_frontage,'LotFrontage'] = y
               
    return cols_added


def add_columns_was_missing(dataset):
    cols_with_missing = [col for col in dataset.columns if dataset[col].isnull().any()]
    new_columns = []
    for col in cols_with_missing:
        new_col = col + '_was_missing'
        new_columns.append(new_col)
        dataset[new_col] = dataset[col].isnull()
    return new_columns

def encode(dataset):
    quality_scale = {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}
    basement_scale = {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}
    access_scale = {"No" : 0, "Grvl" : 1, "Pave" : 2}
    exposure_scale = {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3}
    functional_scale = {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8}
    slope_scale = {"Sev" : 1, "Mod" : 2, "Gtl" : 3}
    shape_scale = {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4}
    paved_scale = {"N" : 0, "P" : 1, "Y" : 2}
    utilities_scale = {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}
    
    replace_table = {"Alley" : access_scale,
                       "BsmtCond" : quality_scale,
                       "BsmtExposure" : exposure_scale,
                       "BsmtFinType1" : basement_scale,
                       "BsmtFinType2" : basement_scale,
                       "BsmtQual" : quality_scale,
                       "ExterCond" : quality_scale,
                       "ExterQual" : quality_scale,
                       "FireplaceQu" : quality_scale,
                       "Functional" : functional_scale,
                       "GarageCond" : quality_scale,
                       "GarageQual" : quality_scale,
                       "HeatingQC" : quality_scale,
                       "KitchenQual" : quality_scale,
                       "LandSlope" : slope_scale,
                       "LotShape" : shape_scale,
                       "PavedDrive" : paved_scale,
                       "PoolQC" : quality_scale,
                       "Street" : access_scale,
                       "Utilities" : utilities_scale}
# Encode some categorical features as ordered numbers when there is information in the order
    mydata = dataset.replace(to_replace = replace_table)
    return (mydata)

def shrink_scales(dataset):
    cols = set(dataset.columns)
    # Create new features
    overall_scale = {1 : 1, 2 : 1, 3 : 1, # bad
                     4 : 2, 5 : 2, 6 : 2, # average
                     7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                     }
    
    quality_scale = {1 : 1, # bad
                     2 : 1, 3 : 1, # average
                     4 : 2, 5 : 2, 6 : 2 # good
                     }
    
    functional_scale = {1 : 1, 2 : 1, # bad
                            3 : 2, 4 : 2, # major
                            5 : 3, 6 : 3, 7 : 3, # minor
                            8 : 4 # typical
                            }
    
    simpl_dict = {}
    simpl_dict["OverallQual"] = overall_scale
    simpl_dict["OverallCond"] = overall_scale
    simpl_dict["PoolQC"] = quality_scale
    simpl_dict["GarageCond"] = quality_scale
    simpl_dict["GarageQual"] = quality_scale
    simpl_dict["FireplaceQu"] = quality_scale
    simpl_dict["Functional"] = functional_scale
    simpl_dict["KitchenQual"] = quality_scale
    simpl_dict["HeatingQC"] = quality_scale
    simpl_dict["BsmtFinType1"] = quality_scale
    simpl_dict["BsmtFinType2"] = quality_scale
    simpl_dict["BsmtCond"] = quality_scale
    simpl_dict["BsmtQual"] = quality_scale
    simpl_dict["ExterCond"] = quality_scale
    simpl_dict["ExterQual"] = quality_scale
    
    new_cols = []
    for (k,v) in simpl_dict.items():
        new_col = "Shrunk_"+ k
        new_cols.append(new_col)
        ds[new_col] = ds[k].replace(v)
    
    return list(set(dataset.columns) - cols)

def add_engineered_features(dataset):
    cols = set(dataset.columns)
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
    
    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
    
    dataset['LowQualFinFrac'] = dataset['LowQualFinSF'] / dataset['GrLivArea']
    dataset['1stFlrFrac'] = dataset['1stFlrSF'] / dataset['GrLivArea']
    dataset['2ndFlrFrac'] = dataset['2ndFlrSF'] / dataset['GrLivArea']
    dataset['TotalAreaSF'] = dataset['GrLivArea'] + dataset['TotalBsmtSF'] + dataset['GarageArea'] + dataset['EnclosedPorch']+dataset['ScreenPorch']
    #dataset['LivingAreaSF'] = dataset['1stFlrSF'] + dataset['2ndFlrSF'] + dataset['BsmtGLQSF'] + dataset['BsmtALQSF'] + dataset['BsmtBLQSF']
    #dataset['StorageAreaSF'] = dataset['LowQualFinSF'] + dataset['BsmtRecSF'] + dataset['BsmtLwQSF'] + dataset['BsmtUnfSF'] + dataset['GarageArea']        

    return list(set(dataset.columns) - cols)
#obj_df["num_cylinders"].value_counts()

def simplify_features(dataset):
    cols = set(dataset.columns)
#    # Simplified overall quality of the house
    dataset["SimplOverallGrade"] = dataset["Shrunk_OverallQual"] * dataset["Shrunk_OverallCond"]
#    # Simplified overall quality of the exterior
    dataset["SimplExterGrade"] = dataset["Shrunk_ExterQual"] * dataset["Shrunk_ExterCond"]
#    # Simplified overall pool score
    dataset["SimplPoolScore"] = dataset["PoolArea"] * dataset["Shrunk_PoolQC"]
#    # Simplified overall garage score
    dataset["SimplGarageScore"] = dataset["GarageArea"] * dataset["Shrunk_GarageQual"]
#    # Simplified overall fireplace score
    dataset["SimplFireplaceScore"] = dataset["Fireplaces"] * dataset["Shrunk_FireplaceQu"]
#    # Simplified overall kitchen score
    dataset["SimplKitchenScore"] = dataset["KitchenAbvGr"] * dataset["Shrunk_KitchenQual"]
    return list(set(dataset.columns) - cols)
    
def polinomial_features(dataset):
    cols = set(dataset.columns)
    
    dataset["OverallQual-s2"] = dataset["OverallQual"] ** 2
    dataset["OverallQual-s3"] = dataset["OverallQual"] ** 3
    dataset["OverallQual-Sq"] = np.sqrt(dataset["OverallQual"])
    dataset["AllSF-2"] = dataset["AllSF"] ** 2
    dataset["AllSF-3"] = dataset["AllSF"] ** 3
    dataset["AllSF-Sq"] = np.sqrt(dataset["AllSF"])
    dataset["AllFlrsSF-2"] = dataset["AllFlrsSF"] ** 2
    dataset["AllFlrsSF-3"] = dataset["AllFlrsSF"] ** 3
    dataset["AllFlrsSF-Sq"] = np.sqrt(dataset["AllFlrsSF"])
    dataset["GrLivArea-2"] = dataset["GrLivArea"] ** 2
    dataset["GrLivArea-3"] = dataset["GrLivArea"] ** 3
    dataset["GrLivArea-Sq"] = np.sqrt(dataset["GrLivArea"])
    dataset["Shrunk_OverallQual-s2"] = dataset["Shrunk_OverallQual"] ** 2
    dataset["Shrunk_OverallQual-s3"] = dataset["Shrunk_OverallQual"] ** 3
    dataset["Shrunk_OverallQual-Sq"] = np.sqrt(dataset["Shrunk_OverallQual"])
    dataset["ExterQual-2"] = dataset["ExterQual"] ** 2
    dataset["ExterQual-3"] = dataset["ExterQual"] ** 3
    dataset["ExterQual-Sq"] = np.sqrt(dataset["ExterQual"])
    dataset["GarageCars-2"] = dataset["GarageCars"] ** 2
    dataset["GarageCars-3"] = dataset["GarageCars"] ** 3
    dataset["GarageCars-Sq"] = np.sqrt(dataset["GarageCars"])
    dataset["TotalBath-2"] = dataset["TotalBath"] ** 2
    dataset["TotalBath-3"] = dataset["TotalBath"] ** 3
    dataset["TotalBath-Sq"] = np.sqrt(dataset["TotalBath"])
    dataset["KitchenQual-2"] = dataset["KitchenQual"] ** 2
    dataset["KitchenQual-3"] = dataset["KitchenQual"] ** 3
    dataset["KitchenQual-Sq"] = np.sqrt(dataset["KitchenQual"])
    dataset["GarageScore-2"] = dataset["GarageScore"] ** 2
    dataset["GarageScore-3"] = dataset["GarageScore"] ** 3
    dataset["GarageScore-Sq"] = np.sqrt(dataset["GarageScore"])
    
    return list(set(dataset.columns) - cols)
    
def score_model(estimator, X, y, n_folds = 5, scoring_func="neg_mean_squared_error"):
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

ds = concat_train_test(train.drop(['SalePrice'], axis=1), test)
#ds = ds.drop(['Utilities'], axis=1)

#ds = ds.drop(high_occurance_missing(ds, 0.8), axis=1)
ds = convert_numeric2category(ds)
was_missing_columns = handle_missing(ds)

ds = encode(ds)
shrunk_columns = shrink_scales(ds)
engineered_columns = add_engineered_features(ds)
simplified_columns = simplify_features(ds)
polinomial_columns = polinomial_features(ds)

num_columns, cat_columns = get_feature_groups(ds)

print("Numerical features : " + str(len(num_columns)))
print("Categorical features : " + str(len(cat_columns)))

predictors = num_columns + cat_columns

engineered_columns = list(set(engineered_columns) - set(cat_columns))

encoded_ds = hot_encode(ds)

log_transform(encoded_ds, list(set(num_columns)-set(was_missing_columns)), 0.5)

train_y = np.log1p(train.SalePrice)
train_X = (encoded_ds.loc[encoded_ds.dataset == "train"]).drop(['dataset', 'Id'], axis=1)
test_X = (encoded_ds.loc[encoded_ds.dataset == "test"]).drop(['dataset', 'Id'], axis=1)

#########################################################################################################
all_predictors = train_X.columns
predictors = list(set(train_X.columns) - set(was_missing_columns))

scaler = RobustScaler()
train_X = scaler.fit(train_X[predictors]).transform(train_X[predictors])
test_X = scaler.transform(test_X[predictors])
train_y = train_y.as_matrix()

sfm = SelectFromModel(model_lasso)
sfm.fit(train_X, train_y)
train_X_reduced = sfm.transform(train_X)
test_X_reduced = sfm.transform(test_X)

#########################################################################################################
hyperparameters_ridge = {'alpha': np.linspace(7.5, 8.5, 25)}
hyperparameters_lasso = {'alpha': np.linspace(0.0001, 0.0009, 10)}
hyperparameters_svr = {'model__C':[1.366, 1.466], 'model__kernel':["rbf"], 
                       "model__gamma":[0.0003, 0.0004], 
                       "model__epsilon":[0.008, 0.009]}

hyperparameters_svr = {'C': np.linspace(1.0, 10, 10), 'epsilon' : np.linspace(0.005, 0.05, 10)}
, C = 1.466, epsilon = 0.0322, gamma = 0.0015
hyperparameters_krr = {'model__alpha':[0.5, 0.6], 'model__kernel':["polynomial"], 'model__degree':[2], 'model__coef0':[2, 2.5]}
hyperparameters_ENet = {'model__alpha':[0.0008, 0.004, 0.0005],'model__l1_ratio':[0.08, 0.1, 0.9],'model__max_iter':[10000]}

hyperparameters_rforest = {"model__n_estimators" : np.arange(230, 250, 5), "model__max_depth": np.arange(10, 16, 2),
                           "model__bootstrap": [True, False]}
hyperparameters_xgb = {"model__booster" : ["gbtree", "gblinear"]}


hyperparameters_rforest = {"model__n_estimators" : [10], "model__max_depth": np.arange(1, 20, 2)}

hyper_models = []
#hyper_models.append((Lasso(), hyperparameters_lasso))
#hyper_models.append((Ridge(), hyperparameters_ridge))
hyper_models.append((SVR(kernel = "rbf", gamma = 0.0004), hyperparameters_svr))
#hyper_models.append((KernelRidge(), hyperparameters_krr))
#hyper_models.append((ElasticNet(), hyperparameters_ENet))
#hyper_models.append((RandomForestRegressor(), hyperparameters_rforest))

for hm, hp in hyper_models:
    hpg = HousePricesGrid(hm, hyperparameters = hp, random = False, random_iter = 20, n_folds = 10)
    hpg.fit(train_X, train_y)
    hpg.get_best_results()
    hpg.plot_scores()

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

models.append(Pipeline([('scaler', RobustScaler()), ('model', model_KRR)]))

names = ["lasso", "ridge", "svr", "ENet", "KRR", "byr"]
 "rforest", "xgb", "GBoost", "lgb"]
#names = ["ridge", "xgb", "lasso", "ENet", "KRR", "GBoost",
#         "lgb", "rforest", "svr", "lsvr", "sgd", "byr", "extra"]
   
scores = [(np.sqrt(score_model(model, train_X_reduced, train_y))).mean() for name, model in zip(names, models)]
tab = pd.DataFrame({ "Model" : names, "Score" : scores })
tab = tab.sort_values(by=['Score'], ascending = True)
print(tab)

for model in models:
    model.fit(train_X, train_y)

averaged_models = AveragingModels(models = (model_lasso, model_ENet, model_svr, model_KRR, 
                                            model_ridge, model_byr) )

score_avg = np.sqrt(score_model(averaged_models, train_X_reduced, train_y))
print(" Averaged base models score: {:.6f} ({:.6f})\n".format(score_avg.mean(), score_avg.std()))

averaged_models.fit(train_X_reduced, train_y)
predicted_prices = np.expm1(averaged_models.predict(test_X_reduced))
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_avg.csv', index=False)


stacked_averaged_models = StackingAveragedModels(base_models = (model_ENet, model_svr, model_KRR, model_lasso,
                                                                model_ridge),
                                                 meta_model = model_byr)

score_stacked_averaged = np.sqrt(score_model(stacked_averaged_models, train_X_reduced, train_y))
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


















def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = "neg_mean_squared_error")
    
    train_scores = np.sqrt(-(train_scores))
    test_scores = np.sqrt(-(test_scores))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Fernando"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
estimator = Pipeline([('scaler', RobustScaler()), ('model', RandomForestRegressor())])
plot_learning_curve(estimator, title, train_X[some_predictors], train_y, (0.0, 0.30), cv=cv)

plt.show()