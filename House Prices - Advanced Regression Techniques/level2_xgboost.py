import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p

DATADIR = "input/"

def score_dataset(train_X, test_X, train_y, test_y):
    forest_model = RandomForestRegressor()
    forest_model.fit(train_X, train_y)
    preds = forest_model.predict(test_X)
    return(mean_absolute_error(test_y, preds))

def read_train_test(train_file = 'train.csv', test_file = 'test.csv'):
    train = pd.read_csv(DATADIR + train_file)
    test = pd.read_csv(DATADIR + test_file)
    return train, test

def concat_train_test(train, test):
    dataset = train.append(test, ignore_index=True)
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

def hot_encode(dataset, drop_list = ['dataset']):
    encoded = pd.get_dummies(dataset.drop(drop_list, axis=1))
    return (pd.merge(encoded, dataset[['Id'] + drop_list], how='inner', on=['Id']))

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
#encoded_ds[skewed_features] = np.log1p(encoded_ds[skewed_features])
#lam = 0.15
#encoded_ds[skewed_features] = boxcox1p(encoded_ds[skewed_features], lam)

train_y = train.SalePrice
train_X = (encoded_ds.loc[encoded_ds.dataset == "train"]).drop(['dataset'], axis=1)
test_X = (encoded_ds.loc[encoded_ds.dataset == "test"]).drop(['dataset'], axis=1)


my_model = XGBRegressor(n_estimators=140)

estimator = Pipeline([("imputer", Imputer()),
                      ("xgb", my_model)])
    
estimator.set_params(imputer__strategy="most_frequent" ,xgb__learning_rate=0.09)
    
score = cross_val_score(estimator, train_X, train_y, scoring='neg_mean_absolute_error').mean()
print(score)

estimator.fit(train_X, train_y)

predicted_prices = estimator.predict(test_X)

print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_xgboost_pipeline_fe.csv', index=False)