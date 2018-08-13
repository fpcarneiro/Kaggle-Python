import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, cross_validate
import seaborn as sns
import matplotlib.pyplot as plt

DATADIR = "input/"

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

def read_train_test(train_file = 'train.csv', test_file = 'test.csv'):
    train = pd.read_csv(DATADIR + train_file)
    test = pd.read_csv(DATADIR + test_file)
    return train, test

def drop_outliers(dataset):
    return(dataset.drop(dataset[(dataset['GrLivArea']>4000) & (dataset['SalePrice']<300000)].index))
    
def fix_testset(dataset):
    dataset_ = dataset.copy()
    dataset_.loc[960, 'PoolQC'] = 'Fa'
    dataset_.loc[1043, 'PoolQC'] = 'Gd'
    dataset_.loc[1139, 'PoolQC'] = 'Fa'

    dataset_.loc[dataset_.GarageYrBlt == 2207, "GarageYrBlt"] = 2007
    return dataset_

def concat_train_test(train, test, ignore_index=False):
    dataset = train.append(test, ignore_index=ignore_index)
    dataset["dataset"] = "train"
    dataset.loc[dataset.Id.isin(test.Id), "dataset"] = "test"
    return dataset

def convert_numeric2category(dataset):
    NumStr = ["MSSubClass", "MoSold"]
    
    for col in NumStr:
        dataset[col]=dataset[col].astype(str)

def high_occurance_missing(dataset, threshold = 0.00000000001):    
    return([c for c in list(dataset.columns) if ( dataset[c].isnull().sum() / len(dataset) ) >= threshold])

def get_feature_groups(dataset):
    num_columns = list(dataset.select_dtypes(exclude=['object', 'category']).columns)
    cat_columns = list(dataset.select_dtypes(include=['object', 'category']).columns)
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
    cols_added = []
    if add_was_missing_columns :
        cols_added = add_columns_was_missing(dataset)
    
    mode_numcols = ['LotFrontage']
    
    no_catcols = ["Alley", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
                  "BsmtFinType2", "BsmtQual", "Fence", "FireplaceQu", 
                  "GarageCond", "GarageFinish", "GarageQual", "GarageType", 
                  "MasVnrType", "MiscFeature", "PoolQC"]
    
    mode_catcols = ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual', 'MSZoning',
                    'SaleType', 'Utilities']
    
    zero_cols = ["MasVnrArea", "BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "GarageArea", "GarageCars"]
    
    cols_mode = mode_numcols + mode_catcols
    
    missing_dict = dict(zip(zero_cols,[0] * len(zero_cols)))
    missing_dict.update(dict(zip(no_catcols,["No"] * len(no_catcols))))
    
    for (k, v) in missing_dict.items():
        dataset.loc[:, k] = dataset.loc[:, k].fillna(v)

    for col in cols_mode:
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
    
    dataset.loc[dataset.GarageYrBlt.isnull(),'GarageYrBlt'] = dataset.loc[dataset.GarageYrBlt.isnull(),'YearBuilt']
               
    return cols_added


def add_columns_was_missing(dataset):
    cols_with_missing = [col for col in dataset.columns if dataset[col].isnull().any()]
    new_columns = []
    for col in cols_with_missing:
        new_col = col + '_was_missing'
        new_columns.append(new_col)
        dataset[new_col] = dataset[col].isnull()
    return new_columns

def encode_features(dataset, features, scales):
    replace_table = dict(zip(features, scales))
    mydata = dataset.replace(to_replace = replace_table)
    return (mydata)

def encode(dataset):
    cols = ["Alley", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual", "ExterCond",
            "ExterQual", "FireplaceQu", "Functional", "GarageCond", "GarageQual", "GarageFinish",
            "HeatingQC", "KitchenQual", "LandSlope", "LotShape", "PavedDrive", "PoolQC", "Street",
            "Utilities", "CentralAir"]
    scales = [access_scale, quality_scale, exposure_scale, basement_scale, basement_scale, quality_scale, 
              quality_scale, quality_scale, quality_scale, functional_scale, quality_scale, quality_scale,
              finished_scale, quality_scale, quality_scale, slope_scale, shape_scale, paved_scale, quality_scale,
              access_scale, utilities_scale, air_scale]

    mydata = encode_features(dataset, cols, scales)
    return (mydata)

def shrink_scales(dataset, prefix = ""):
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
        new_col = prefix + k
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
    dataset["TotalBath"] = dataset["BsmtFullBath"] + (0.5 * dataset["BsmtHalfBath"]) + dataset["FullBath"] + (0.5 * dataset["HalfBath"])
#    # Total SF for house (incl. basement)
    dataset["AllSF"] = dataset["GrLivArea"] + dataset["TotalBsmtSF"]
#    # Total SF for 1st + 2nd floors
    dataset["AllFlrsSF"] = dataset["1stFlrSF"] + dataset["2ndFlrSF"]
#    # Total SF for porch
    dataset["AllPorchSF"] = dataset["OpenPorchSF"] + dataset["EnclosedPorch"] + dataset["3SsnPorch"] + dataset["ScreenPorch"]
## Has masonry veneer or not
    dataset["HasMasVnr"] = dataset.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, "Stone" : 1, "None" : 0})
#    # House completed before sale or not
    dataset["BoughtOffPlan"] = dataset.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
    
    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
    
    dataset['LowQualFinFrac'] = dataset['LowQualFinSF'] / dataset['GrLivArea']
    dataset['1stFlrFrac'] = dataset['1stFlrSF'] / dataset['GrLivArea']
    dataset['2ndFlrFrac'] = dataset['2ndFlrSF'] / dataset['GrLivArea']
    dataset['TotalAreaSF'] = dataset['GrLivArea'] + dataset['TotalBsmtSF'] + dataset['GarageArea'] + dataset['EnclosedPorch']+dataset['ScreenPorch']
    #dataset['LivingAreaSF'] = dataset['1stFlrSF'] + dataset['2ndFlrSF'] + dataset['BsmtGLQSF'] + dataset['BsmtALQSF'] + dataset['BsmtBLQSF']
    #dataset['StorageAreaSF'] = dataset['LowQualFinSF'] + dataset['BsmtRecSF'] + dataset['BsmtLwQSF'] + dataset['BsmtUnfSF'] + dataset['GarageArea']        

    return list(set(dataset.columns) - cols)
#obj_df["num_cylinders"].value_counts()

def correlation_target(dataset, target = "SalePrice"):
    corr = dataset.corr()
    corr.sort_values([target], ascending = False, inplace = True)
    return(corr.SalePrice)
    
def correlation_matrix(dataset, target = 'SalePrice', nvar = 10):
    corrmat = dataset.corr()
    cols = corrmat.nlargest(nvar + 1, target)[target].index
    cm = np.corrcoef(dataset[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                     annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return list(cols[1:])

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
    
def polinomial_features(dataset, polinomial_cols = ["OverallQual", "AllSF", "AllFlrsSF", "GrLivArea", "Shrunk_OverallQual",
            "ExterQual", "GarageCars", "TotalBath", "KitchenQual", "GarageScore"]):
    cols = set(dataset.columns)
    
    for col in polinomial_cols:
        dataset[col+'-2'] = dataset.loc[:,col]**2
        dataset[col+'-3'] = dataset.loc[:,col]**3
        dataset[col+'-sqrt'] = np.sqrt(np.absolute(dataset.loc[:,col]))
    
    return list(set(dataset.columns) - cols)

def score_model(estimator, X, y, n_folds = 5, scoring_func="neg_mean_squared_error"):
    kf = KFold(n_folds, shuffle=True).get_n_splits(X)
    score = -cross_val_score(estimator, X, y, scoring=scoring_func, cv = kf)
    return(score)

def log_transform(dataset, cols = None, threshold = 0.5):
    if cols == None:
        cols = list(dataset.select_dtypes(exclude=['object', 'category']).columns)
    skewness = dataset[cols].apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > threshold]
    #print(str(skewness.shape[0]) + " skewed numerical features to log transform")
    skewed_features = skewness.index
    #dataset[skewed_features] = np.log1p(dataset[skewed_features])
    return list((dataset[skewed_features]).columns)
    
def more_features(dataset):
    cols = set(dataset.columns)
    dataset["IsRegularLotShape"] = (dataset["LotShape"] == "Reg") * 1
    # Most properties are level; bin the other possibilities together
    # as "not level".
    dataset["IsLandLevel"] = (dataset["LandContour"] == "Lvl") * 1
    # Most land slopes are gentle; treat the others as "not gentle".
    dataset["IsLandSlopeGentle"] = (dataset["LandSlope"] == "Gtl") * 1
    # Most properties use standard circuit breakers.
    dataset["IsElectricalSBrkr"] = (dataset["Electrical"] == "SBrkr") * 1
    # About 2/3rd have an attached garage.
    dataset["IsGarageDetached"] = (dataset["GarageType"] == "Detchd") * 1
    # Most have a paved drive. Treat dirt/gravel and partial pavement
    # as "not paved".
    dataset["IsPavedDrive"] = (dataset["PavedDrive"] == "Y") * 1
    # The only interesting "misc. feature" is the presence of a shed.
    #dataset["HasShed"] = (dataset["MiscFeature"] == "Shed") * 1.
           
    dataset["Remodeled"] = (dataset["YearRemodAdd"] != dataset["YearBuilt"]) * 1
    
    # Did a remodeling happen in the year the house was sold?
    dataset["RecentRemodel"] = (dataset["YearRemodAdd"] == dataset["YrSold"]) * 1
    
    # Was this house sold in the year it was built?
    dataset["VeryNewHouse"] = (dataset["YearBuilt"] == dataset["YrSold"]) * 1
    return list(set(dataset.columns) - cols)

def have_stuff_features(dataset):
    cols = set(dataset.columns)
    dataset['HasBsmt'] = 0 
    dataset["HasShed"] = 0
    dataset["Has2ndFloor"] = 0
    dataset["HasMasVnr"] = 0
    dataset["HasWoodDeck"] = 0
    dataset["HasOpenPorch"] = 0
    dataset["HasEnclosedPorch"] = 0
    dataset["Has3SsnPorch"] = 0
    dataset["HasScreenPorch"] = 0
    
    dataset.loc[dataset['TotalBsmtSF'] != 0,'HasBsmt'] = 1
    dataset.loc[dataset["MiscFeature"] == "Shed","HasShed"] = 1
    dataset.loc[dataset["2ndFlrSF"] != 0,"Has2ndFloor"] = 1
    dataset.loc[dataset["MasVnrArea"] != 0,"HasMasVnr"] = 1
    dataset.loc[dataset["WoodDeckSF"] != 0,"HasWoodDeck"] = 1
    dataset.loc[dataset["OpenPorchSF"] != 0,"HasOpenPorch"] = 1
    dataset.loc[dataset["EnclosedPorch"] != 0,"HasEnclosedPorch"] = 1
    dataset.loc[dataset["3SsnPorch"] != 0,"Has3SsnPorch"] = 1
    dataset.loc[dataset["ScreenPorch"] != 0,"HasScreenPorch"] = 1
    
    return list(set(dataset.columns) - cols)

def add_features_neighborhood(dataset):
    aggregations = {
            'SalePrice_Median':'median',
            'SalePrice_Mean':'mean',
            'SalePrice_Std':'std'
            }
    
    dataset["Neighborhood_SalePrice_Median"] = dataset.groupby("Neighborhood")["SalePrice"].transform(lambda x: x.median())
    #dataset["Neighborhood_SalePrice_Mean"] = dataset.groupby("Neighborhood")["SalePrice"].transform(lambda x: x.mean())
    #dataset["Neighborhood_SalePrice_by_SF"] = dataset.groupby("Neighborhood")["SalePrice"].transform(lambda x: x.mean())
    #train.groupby("Neighborhood")["SalePrice"].agg(aggregations)

def cross_validate_model(estimator, X, y, n_folds = 5, repetitions = 1, scoring_func="neg_mean_squared_error", 
                         seed = 2018, return_train_score = False, jobs = -1):
    if repetitions == 1:
        kf = KFold(n_splits = n_folds, shuffle=True, random_state = seed)
    else:
        kf = RepeatedKFold(n_splits = n_folds, n_repeats = repetitions, random_state = seed)
    scores = cross_validate(estimator, X, y, scoring=scoring_func, cv = kf, return_train_score = return_train_score, n_jobs = jobs)
    return(scores)

model_name = "Estimator"
score_mean = "Score (mean)"
score_std = "Score (std)"
scores_field = "CV Scores"

def get_cross_validate(medels_list, X, y, folds = 5, repetitions = 1, seed = 2018, train_score = False, jobs = -1): 
    sort_by = "Score (mean)"
    if train_score:
        results = pd.DataFrame(columns = [model_name, score_mean, score_std, scores_field,
                                          "Train Score (mean)", "Train Score (std)", "Train CV Scores"])
    else:
        results = pd.DataFrame(columns = [model_name, score_mean, score_std, scores_field])
        
    for name, model in medels_list:
        scores =  cross_validate_model(model, X, y, n_folds = folds, repetitions = repetitions, seed = seed, 
                                       return_train_score = train_score, jobs = jobs)
        test_score = np.sqrt(-scores["test_score"])
        record = {model_name: name,
                  score_mean: test_score.mean(),
                  score_std: test_score.std(),
                  scores_field: test_score}
        
        if train_score:
            train_score = np.sqrt(-scores["train_score"])
            record["Train Score (mean)"] = train_score.mean()
            record["Train Score (std)"] = train_score.std()
            record["Train CV Scores"] = train_score
        
        results = results.append(record, ignore_index=True)
    results.sort_values(by=[sort_by], ascending = True, inplace = True)
    return results

def get_submission_file_name(results_table):
    template = '{0}_cv_{1:.4f}_std_{2:.4f}.csv'
    return (template.format(results_table.loc[0,:][model_name], results_table.loc[0,:][score_mean], 
                    results_table.loc[0,:][score_std]))

def get_splits_year(dataset, column):
    vc = list(dataset[column].value_counts(sort=False).index.values)
    for year in vc[:-1]:
        train_index = dataset[dataset[column] <= year].index.values
        test_index = dataset[dataset[column] > year].index.values
        yield (train_index, test_index)

def get_constant_features(dataset):
    feats_counts = dataset.nunique(dropna = False)
    constant_features = feats_counts.loc[feats_counts==1].index.tolist()
    return constant_features

def equal_columns(col_a, col_b):
    return np.all(col_a == col_b)

from tqdm import tqdm
def duplicate_columns(df, return_dataframe = False, verbose = False, progress = True):
    '''
        a function to detect and possibly remove duplicated columns for a pandas dataframe
    '''
    # group columns by dtypes, only the columns of the same dtypes can be duplicate of each other
    groups = df.columns.to_series().groupby(df.dtypes).groups
    duplicated_columns = {}
    

        
    for dtype, col_names in groups.items():
        column_values = df[col_names]
        num_columns = len(col_names)
        
        if progress == True:
            it = tqdm(range(num_columns))
        else:
            it = range(num_columns)
        # find duplicated columns by checking pairs of columns, store first column name if duplicate exist 
        for i in it:
            column_i = column_values.iloc[:,i]
            for j in range(i + 1, num_columns):
                column_j = column_values.iloc[:,j]
                if equal_columns(column_i, column_j):
                    if verbose: 
                        print("column {} is a duplicate of column {}".format(col_names[i], col_names[j]))
                    duplicated_columns[col_names[j]] = col_names[i]
                    break
    if not return_dataframe:
        # return the column names of those duplicated exists
        return duplicated_columns
    else:
        # return a dataframe with duplicated columns dropped 
        return df.drop(labels = list(duplicated_columns.keys()), axis = 1)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_processed_datasets():
    train, test = read_train_test()

    train = drop_outliers(train)
    
    train_y = (np.log1p(train.SalePrice)).values
    
    all_data = (concat_train_test(train, test)).drop(['Id'], axis=1)
    
    #add_features_neighborhood(all_data)
    
    all_data.drop(['SalePrice'], axis=1, inplace = True)
    
    convert_numeric2category(all_data)
    was_missing_columns = handle_missing(all_data, False)
    
    even_more_features = more_features(all_data)
    
    all_data = encode(all_data)
    shrunk_columns = shrink_scales(all_data, "Shrunk_")
    
    engineered_columns = add_engineered_features(all_data)
    simplified_columns = simplify_features(all_data)
    polinomial_columns = polinomial_features(all_data)
    
    num_columns, cat_columns = get_feature_groups(all_data, drop_list = ['dataset'])
    
    print("Numerical features : " + str(len(num_columns)))
    print("Categorical features : " + str(len(cat_columns)))
    
    engineered_columns = list(set(engineered_columns) - set(cat_columns))
    
    have_features = have_stuff_features(all_data)
    
    all_data_encoded = hot_encode(all_data, drop_list = ['dataset'])
    
    skewed_features = log_transform(all_data_encoded, list(set(num_columns)-set(was_missing_columns)), 0.5)
    
    #train_y = (np.log1p(train.SalePrice)).values
    train_X = (all_data_encoded.loc[all_data_encoded.dataset == "train"]).drop(['dataset'], axis=1)
    test_X = (all_data_encoded.loc[all_data_encoded.dataset == "test"]).drop(['dataset'], axis=1)
    
    return (train_X, train_y, test_X, list(test.Id))




