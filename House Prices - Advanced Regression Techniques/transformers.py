import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew

quality_scale = {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}
basement_scale = {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}
access_scale = {"No" : 0, "Grvl" : 1, "Pave" : 2}
exposure_scale = {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3}
functional_scale = {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8}
slope_scale = {"Sev" : 1, "Mod" : 2, "Gtl" : 3}
shape_scale = {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4}
paved_scale = {"N" : 0, "P" : 1, "Y" : 2}
utilities_scale = {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}
air_scale = {"N" : 0, "Y" : 1}
finished_scale = {"No" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3}

def add_date_related_features(X):
    X_ = X.copy()
    X_['Age'] = X_['YrSold'] - X_['YearBuilt']
    X_.loc[X_.YearBuilt > X_.YrSold, "Age"] = 0
    X_['RemodeledAge'] = X_['YrSold'] - X_['YearRemodAdd']
    X_.loc[X_.YearRemodAdd > X_.YrSold, "RemodeledAge"] = 0
    X_['GarageAge'] = X_['YrSold'] - X_['GarageYrBlt']
    X_.loc[X_.GarageYrBlt > X_.YrSold, "GarageAge"] = 0
    X_.loc[:, "IsHighSeason"] = (X_.loc[:, "MoSold"].isin([5, 6, 7]))
    return X_

def get_feature_groups(X):
    num_columns = list(X.select_dtypes(exclude=['object']).columns)
    cat_columns = list(X.select_dtypes(include=['object']).columns)
    return (num_columns, cat_columns)

def drop_outliers(X):
    return(X.drop(X[(X['GrLivArea']>4000) & (X['SalePrice']<300000)].index))

def convert2category(X, columns = None):
    X.loc[:, columns] = X.loc[:, columns].astype('category')
    return X

def more_features(X):
    X.loc[:, "IsRegularLotShape"] = (X.loc[:, "LotShape"] == "Reg") * 1
    X.loc[:, "IsLandLevel"] = (X.loc[:, "LandContour"] == "Lvl") * 1
    #X.loc[:, "IsLandSlopeGentle"] = (X.loc[:, "LandSlope"] == "Gtl") * 1       
    #X.loc[:, "IsElectricalSBrkr"] = (X.loc[:, "Electrical"] == "SBrkr") * 1
    #X.loc[:, "IsGarageDetached"] = (X.loc[:, "GarageType"] == "Detchd") * 1
    X.loc[:, "IsPavedDrive"] = (X.loc[:, "PavedDrive"] == "Y") * 1    
    return X

def add_columns_was_missing(X):
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    new_columns = []
    for col in cols_with_missing:
        new_col = col + '_was_missing'
        new_columns.append(new_col)
        X[new_col] = X[col].isnull().astype(int)
    return X

def handle_missing(X, add_was_missing_columns = False):
    group_by_col = "Neighborhood"
    
    if add_was_missing_columns :
        X = add_columns_was_missing(X)
    
    median_numcols = ['LotFrontage']
    
    no_catcols = ["Alley", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
                  "BsmtFinType2", "BsmtQual", "Fence", "FireplaceQu", 
                  "GarageCond", "GarageFinish", "GarageQual", "GarageType", 
                  "MasVnrType", "MiscFeature", "PoolQC"]
    
    mode_catcols = ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual', 'MSZoning',
                    'SaleType', 'Utilities']
    
    zero_cols = ["MasVnrArea", "BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "GarageArea", "GarageCars"]
    
    cols_mode = mode_catcols
    
    missing_dict = dict(zip(zero_cols,[0] * len(zero_cols)))
    missing_dict.update(dict(zip(no_catcols,["No"] * len(no_catcols))))
    
    for (k, v) in missing_dict.items():
        X.loc[:, k] = X.loc[:, k].fillna(v)

    for col in cols_mode:
        if X[col].isnull().any():
            X[col] = X.groupby(group_by_col)[col].transform(lambda x: x.fillna(x.mode()[0]))
        
    for col in median_numcols:
        if X[col].isnull().any():
            X[col] = X.groupby(group_by_col)[col].transform(lambda x: x.fillna(x.median()))
    
    X.loc[X.GarageYrBlt.isnull(),'GarageYrBlt'] = X.loc[X.GarageYrBlt.isnull(),'YearBuilt']
    
    return X

def encode_features(X, features, scales):
    replace_table = dict(zip(features, scales))
    X.replace(to_replace = replace_table, inplace = True)
    return X

def encode(X):
#    cols = ["Alley", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual", "ExterCond",
#            "ExterQual", "FireplaceQu", "Functional", "GarageCond", "GarageQual", "GarageFinish",
#            "HeatingQC", "KitchenQual", "LandSlope", "LotShape", "PavedDrive", "PoolQC", "Street",
#            "Utilities", "CentralAir"]
#    scales = [access_scale, quality_scale, exposure_scale, basement_scale, basement_scale, quality_scale, quality_scale,
#              quality_scale, quality_scale, functional_scale, quality_scale, quality_scale, finished_scale, 
#              quality_scale, quality_scale, slope_scale, shape_scale, paved_scale, quality_scale, access_scale, 
#              utilities_scale, air_scale]
    
    #cols_quality = ["BsmtCond", "BsmtQual", "ExterCond", "ExterQual", "FireplaceQu", "GarageCond", "GarageQual",
    #                "HeatingQC", "KitchenQual", "PoolQC"]
    
    cols_quality = ["GarageCond", "GarageQual", "ExterCond", "ExterQual", "BsmtCond", "BsmtQual"]

    scales = [quality_scale] * len(cols_quality)
    
    return encode_features(X, cols_quality, scales)

def shrink_scales(X, prefix = ""):
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
    
    for (k,v) in simpl_dict.items():
        new_col = prefix + k
        X[new_col] = X[k].replace(v)
    
    return X

def add_engineered_features(X):
    X["OverallGrade"] = X["OverallQual"] * X["OverallCond"]
    #X["GarageGrade"] = X["GarageQual"] * X["GarageCond"]
    #X["ExterGrade"] = X["ExterQual"] * X["ExterCond"]
    #X["BsmtGrade"] = X["BsmtQual"] * X["BsmtCond"]
    #X["KitchenScore"] = X["KitchenAbvGr"] * X["KitchenQual"]
    #X["FireplaceScore"] = X["Fireplaces"] * X["FireplaceQu"]
    #X["GarageScore"] = X["GarageArea"] * X["GarageQual"]
    #X["PoolScore"] = X["PoolArea"] * X["PoolQC"]
    #X["TotalBath"] = X["BsmtFullBath"] + (0.5 * X["BsmtHalfBath"]) + X["FullBath"] + (0.5 * X["HalfBath"])
    X["AllSF"] = X["GrLivArea"] + X["TotalBsmtSF"]
    X["AllFlrsSF"] = X["1stFlrSF"] + X["2ndFlrSF"]
    X["AllPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
    #X["HasMasVnr"] = X.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, "Stone" : 1, "None" : 0})
    #X["BoughtOffPlan"] = X.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
    X['LowQualFinFrac'] = X['LowQualFinSF'] / X['GrLivArea']
    #X['1stFlrFrac'] = X['1stFlrSF'] / X['GrLivArea']
    #X['2ndFlrFrac'] = X['2ndFlrSF'] / X['GrLivArea']
    X['TotalAreaSF'] = X['GrLivArea'] + X['TotalBsmtSF'] + X['GarageArea'] + X['EnclosedPorch']+X['ScreenPorch']
    
    #X['LivingAreaSF'] = X['1stFlrSF'] + X['2ndFlrSF'] + X['BsmtGLQSF'] + X['BsmtALQSF'] + X['BsmtBLQSF']
    #X['StorageAreaSF'] = X['LowQualFinSF'] + X['BsmtRecSF'] + X['BsmtLwQSF'] + X['BsmtUnfSF'] + X['GarageArea']
    return X

def add_simplified_features(X, prefix = ""):
    X["SimplOverallGrade"] = X[prefix + "OverallQual"] * X[prefix + "OverallCond"]
    X["SimplExterGrade"] = X[prefix + "ExterQual"] * X[prefix + "ExterCond"]
    X["SimplPoolScore"] = X["PoolArea"] * X[prefix + "PoolQC"]
    X["SimplGarageScore"] = X["GarageArea"] * X[prefix + "GarageQual"]
    X["SimplFireplaceScore"] = X["Fireplaces"] * X[prefix + "FireplaceQu"]
    X["SimplKitchenScore"] = X["KitchenAbvGr"] * X[prefix + "KitchenQual"]
    return X

def add_polinomial_features(X, polinomial_cols = ["OverallQual", "AllSF", "AllFlrsSF", "GrLivArea", "Shrunk_OverallQual",
            "ExterQual", "GarageCars", "TotalBath", "KitchenQual", "GarageScore"]):
    
    for col in polinomial_cols:
        X[col+'-2'] = X.loc[:,col]**2
        #X[col+'-3'] = X.loc[:,col]**3
        #X[col+'-sqrt'] = np.sqrt(np.absolute(X.loc[:,col]))
    
    return X

def have_stuff_features(X):
    X['HasBsmt'] = 0 
    X["HasShed"] = 0
    X["Has2ndFloor"] = 0
    X["HasMasVnr"] = 0
    X["HasWoodDeck"] = 0
    X["HasOpenPorch"] = 0
    X["HasEnclosedPorch"] = 0
    X["Has3SsnPorch"] = 0
    X["HasScreenPorch"] = 0
    
    X.loc[X['TotalBsmtSF'] != 0,'HasBsmt'] = 1
    X.loc[X["MiscFeature"] == "Shed","HasShed"] = 1
    X.loc[X["2ndFlrSF"] != 0,"Has2ndFloor"] = 1
    X.loc[X["MasVnrArea"] != 0,"HasMasVnr"] = 1
    X.loc[X["WoodDeckSF"] != 0,"HasWoodDeck"] = 1
    X.loc[X["OpenPorchSF"] != 0,"HasOpenPorch"] = 1
    X.loc[X["EnclosedPorch"] != 0,"HasEnclosedPorch"] = 1
    X.loc[X["3SsnPorch"] != 0,"Has3SsnPorch"] = 1
    X.loc[X["ScreenPorch"] != 0,"HasScreenPorch"] = 1
    
    return X

def add_neighbourhood_related_features(X, cols = None):
    group_by_col = "Neighborhood"
    
    if cols == None:
        #cols = ["GrLivArea", "OverallQual", "YearBuilt", "Age", "BsmtFinSF1"]
        num_columns = list(X.select_dtypes(exclude=['object']).columns)
        cat_columns = list(X.select_dtypes(include=['object']).columns)

    for col in num_columns:
        new_col = group_by_col + "_" + col + "_" + "Mean"
        X[new_col] = X.groupby(group_by_col)[col].transform(lambda x: x.mean())
        new_col = group_by_col + "_" + col + "_" + "Median"
        X[new_col] = X.groupby(group_by_col)[col].transform(lambda x: x.median())
        new_col = group_by_col + "_" + col + "_" + "Max"
        X[new_col] = X.groupby(group_by_col)[col].transform(lambda x: x.max())
        new_col = group_by_col + "_" + col + "_" + "Min"
        X[new_col] = X.groupby(group_by_col)[col].transform(lambda x: x.min())
    
    return X

def hot_encode(X):
    return pd.get_dummies(X)
    #return (pd.concat([X, encoded], axis=1).drop(columns, axis=1))

class DateRelatedFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return add_date_related_features(X)

    def fit(self, X, y=None):
        return self
    
class NeighbourhoodRelatedFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return add_neighbourhood_related_features(X)

    def fit(self, X, y=None):
        return self

class DropOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return drop_outliers(X)

    def fit(self, X, y=None):
        return self

class Convert2CategoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns

    def transform(self, X, y=None):
        return convert2category(X, self.columns)

    def fit(self, X, y=None):
        if self.columns == None:
            self.columns = list(X.select_dtypes(include=['object']).columns)
        return self

class HandleMissingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, was_missing_features = False):
        self.was_missing_features = was_missing_features

    def transform(self, X, y=None):
        return handle_missing(X, self.was_missing_features)

    def fit(self, X, y=None):
        return self
 
class MoreFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return more_features(X)

    def fit(self, X, y=None):
        return self
    
class EncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, prefix = "Shrunk_"):
        self.prefix = prefix

    def transform(self, X, y=None):
        return encode(X)

    def fit(self, X, y=None):
        return self

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return add_engineered_features(X)

    def fit(self, X, y=None):
        return self

class SimplifiedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, prefix = "Shrunk_"):
        self.prefix = prefix

    def transform(self, X, y=None):
        return add_simplified_features(X, self.prefix)

    def fit(self, X, y=None):
        return self
    
class PolinomialFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, polinomial_cols = None, col = "SalePrice"):
        self.column = col
        self.polinomial_cols = polinomial_cols

    def transform(self, X, y=None):
        return add_polinomial_features(X, self.polinomial_cols)

    def fit(self, X, y=None):
        if self.polinomial_cols == None:
            corr = X.corr().loc[:, self.column]
            corr = corr.sort_values(ascending = False)
            self.polinomial_cols = list(corr[1:11].index)
        return self

class HaveStuffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return have_stuff_features(X)

    def fit(self, X, y=None):
        return self

class HotEncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return hot_encode(X)

    def fit(self, X, y=None):
        return self
    
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        X = np.log1p(X)
        return X

    def fit(self, X, y=None):
        print(X.shape[1])
        return self
    
class TypeSelectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
    
class ColumnsSelectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns = [], exclude = False):
        self.columns = columns
        self.exclude = exclude
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if self.exclude:
            return X.drop(self.columns, axis=1)
        else:
            return X[self.columns]
    
class StringIndexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))