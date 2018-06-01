import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

DATADIR = "input/"

def read_train_test(train_file = 'train.csv', test_file = 'test.csv'):
    train = pd.read_csv(DATADIR + train_file)
    test = pd.read_csv(DATADIR + test_file)
    return train, test

def drop_outliers(dataset):
    return(dataset.drop(dataset[(dataset['GrLivArea']>4000) & (dataset['SalePrice']<300000)].index))

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
        cols_added = add_columns_was_missing(dataset)
    
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
        dataset[new_col] = dataset[k].replace(v)
    
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