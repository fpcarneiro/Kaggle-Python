import gc
import feature_selection as fs
import preprocessing as pp
from preprocessing import timer
import load as ld
from training import display_importances, get_folds, save_importances, AveragingModels
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier, Dataset, cv, train
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split

def get_tree_models():
    lgb_params = {}
    lgb_params['nthread'] = 2
    lgb_params['n_estimators'] = 10000
    lgb_params['learning_rate'] = 0.02
    lgb_params['num_leaves'] = 34
    lgb_params['colsample_bytree'] = 0.5
    lgb_params['subsample'] = 0.7379
    lgb_params['max_depth'] = 8
    lgb_params["reg_alpha"] = 0.041545473
    lgb_params['reg_lambda'] = 0.0735294
    lgb_params['min_split_gain'] = 0.0735294
    lgb_params['min_child_weight'] = 0.0735294
    lgb_params['silent'] = False
    
    xgb_params = dict()
    xgb_params["booster"] = "gbtree"
    xgb_params["objective"] = "binary:logistic"
    xgb_params["n_estimators"] = 10000
    xgb_params["colsample_bytree"] = 0.4385
    xgb_params["subsample"] = 0.7379
    xgb_params["max_depth"] = 3
    xgb_params['reg_alpha'] = 0.041545473
    xgb_params['reg_lambda'] = 0.0735294
    xgb_params["learning_rate"] = 0.02
    xgb_params["min_child_weight"] = 2
    xgb_params['silent'] = False

    tree_models = []
    for seed in [2017]:
        lgb_params['seed'] = seed
        xgb_params['seed'] = seed
        
        lgbm = LGBMClassifier(**lgb_params)
        xgb = XGBClassifier(**xgb_params)
        tree_models.append(("LightGBM_" + str(seed), lgbm))
        tree_models.append(("XGBoost_" + str(seed), xgb))
    return tree_models

def get_importances_from_model(X, y, features = None, verbose = 50, early_stopping_rounds = 200):
     
    lgb_params = {}
    lgb_params['boosting_type'] = 'gbdt'
    lgb_params['objective'] = 'binary'
    lgb_params['learning_rate'] = 0.02
    lgb_params['metric'] = 'auc'
#    lgb_params['num_leaves'] = 34
    lgb_params['colsample_bytree'] = 0.75
    lgb_params['subsample'] = 0.75
    lgb_params['n_estimators'] = 10000
#    lgb_params['max_depth'] = 8
#    lgb_params["reg_alpha"] = 0.041545473
#    lgb_params['reg_lambda'] = 0.0735294
#    lgb_params['min_split_gain'] = 0.0735294
#    lgb_params['min_child_weight'] = 0.0735294
#    lgb_params['silent'] = False
    
    if features == None:
        features = X.columns.tolist()
        
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 2017)
    
    lgb_train = Dataset(data = train_X, label = train_y, feature_name = features)
    lgb_val = Dataset(data = val_X, label = val_y, feature_name = features)
    
    lgb_booster = train(params = lgb_params, train_set = lgb_train, valid_sets = [lgb_train, lgb_val], valid_names = ["train", "validation"], 
            verbose_eval = verbose, early_stopping_rounds = early_stopping_rounds)

    return lgb_booster

def get_datasets(debug_size, silent, treat_duplicated = True):
    train, test = ld.get_processed_files(debug_size, silent)
    
    train = pp.convert_types(train, print_info = not silent)
    test = pp.convert_types(test, print_info = not silent)
    
    features = [f for f in train.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    
    if treat_duplicated:
        with timer("Treating Duplicated"):
            duplicated = pp.duplicate_columns(train, verbose = True, progress = False)
            if len(duplicated) > 0:
                train.drop(list(duplicated.keys()), axis=1, inplace = True)
                test.drop(list(duplicated.keys()), axis=1, inplace = True)
    
    train_y = train['TARGET']
    train_X = train.loc[:, features]

    ids = test['SK_ID_CURR']
    test_X = test.loc[:, features]
    
    return train_X, train_y, test_X, ids

if __name__ == "__main__":
    debug_size = 0
    silent = True
    verbose = 50
    early_stopping_rounds = 200
    
    with timer("Full Model Run"):
        
        train_X, train_y, test_X, ids = get_datasets(debug_size = debug_size, silent = silent)
        
        lgb_booster = get_importances_from_model(train_X, train_y, train_X.columns.tolist(), verbose = verbose,  early_stopping_rounds = early_stopping_rounds)
        
        importances_booster = save_importances(train_X.columns.tolist(), lgb_booster.feature_importance(), sort= True, drop_importance_zero = True)
        
        feats = importances_booster.FEATURE.tolist()
        
        for name, m in get_tree_models():
        
            with timer("Run " + name):
                
                for i in range(50,151,50):
                    
                    model = AveragingModels(m)
                    
                    print(feats[:i])
                
                    model.fit(train_X.loc[:, feats[:i]], train_y, folds = 5, stratified = False, verbose = verbose, early_stopping_rounds = early_stopping_rounds)
                    pred = model.predict_proba(test_X.loc[:, feats[:i]])
                    
                    cv_score = model.auc_score
                    feat_importance = model.importances_df
                    
                    #display_importances(feat_importance)
                    submission = pp.submit_file(ids, pred, prefix_file_name = name, cv_score = cv_score)
                    
                    del model, pred, cv_score, feat_importance, submission
                    gc.collect()
            
    del test_X, train_X, train_y
    del features_variance
    gc.collect()
