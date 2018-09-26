import gc
import preprocessing as pp
from preprocessing import timer
import load as ld
from training import save_importances, AveragingModels

from sklearn.feature_selection import VarianceThreshold

import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier, Dataset, train
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn import ensemble

def get_tree_models():
    lgb_params = {}
    lgb_params['nthread'] = 2
    lgb_params['n_estimators'] = 10000
    lgb_params['learning_rate'] = 0.01
    #lgb_params['colsample_bytree'] = 0.5
    #lgb_params['subsample'] = 0.8
#    lgb_params['max_depth'] = 8
#    lgb_params["reg_alpha"] = 0.041545473
#    lgb_params['reg_lambda'] = 0.0735294
#    lgb_params['min_split_gain'] = 0.0735294
#    lgb_params['min_child_weight'] = 0.0735294
#    lgb_params['num_leaves'] = 34
    lgb_params['silent'] = False
    
    xgb_params = dict()
    xgb_params["booster"] = "gbtree"
    xgb_params["objective"] = "binary:logistic"
    xgb_params["n_estimators"] = 10000
    #xgb_params["colsample_bytree"] = 0.5
    #xgb_params["subsample"] = 0.8
#    xgb_params["max_depth"] = 3
#    xgb_params['reg_alpha'] = 0.041545473
#    xgb_params['reg_lambda'] = 0.0735294
    xgb_params["learning_rate"] = 0.01
#    xgb_params["min_child_weight"] = 2
    xgb_params['silent'] = False
    
    lgb_fit_params = {}
    lgb_fit_params['eval_metric'] = 'auc'
    lgb_fit_params['verbose'] = 0
    lgb_fit_params['early_stopping_rounds'] = 200
    lgb_fit_params['eval_set'] = {}
    lgb_fit_params['eval_names'] = ["train", "validation"]
    
    xgb_fit_params = {}
    xgb_fit_params['eval_metric'] = 'auc'
    xgb_fit_params['verbose'] = 0
    xgb_fit_params['early_stopping_rounds'] = 200
    xgb_fit_params['eval_set'] = {}
    
    params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    clf = ensemble.GradientBoostingClassifier(**params)

    tree_models = []
    for seed in [2017]:
        lgb_params['seed'] = seed
        xgb_params['seed'] = seed
        
        lgbm = LGBMClassifier(**lgb_params)
        xgb = XGBClassifier(**xgb_params)
        tree_models.append(("GBoost_" + str(seed), clf, {}))
        tree_models.append(("LightGBM_" + str(seed), lgbm, lgb_fit_params))
        tree_models.append(("XGBoost_" + str(seed), xgb, xgb_fit_params))
        
    return tree_models

def get_importances_from_model(X, y, features = None, verbose = 50, early_stopping_rounds = 200):
    
    lgb_params = {}
    lgb_params['boosting_type'] = 'gbdt'
    lgb_params['objective'] = 'binary'
    lgb_params['learning_rate'] = 0.03
    lgb_params['metric'] = 'auc'
    lgb_params['num_iterations'] = 10000
    lgb_params["colsample_bytree"] = 0.5
    lgb_params["subsample"] = 0.8
    lgb_params["reg_alpha"] = 0.3
    lgb_params['reg_lambda'] = 0.3
    lgb_params['max_depth'] = 8
       
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
        
        features = train_X.columns.tolist()
        
        lgb_booster = get_importances_from_model(train_X, train_y, features, verbose = verbose,  early_stopping_rounds = early_stopping_rounds)
        
        importances_booster = save_importances(train_X.columns.tolist(), lgb_booster.feature_importance(), sort= True, drop_importance_zero = True)
        
        threshold = importances_booster.IMPORTANCE.median()/2
        
        feats = importances_booster[importances_booster.IMPORTANCE > threshold].FEATURE.tolist()
        
        #scaler = MinMaxScaler()
        #train_X = scaler.fit_transform(train_X)
        #test_X = scaler.fit(test_X)
        
        selector = SelectFromModel(estimator = lgb_booster, threshold = "median", prefit = True)
        selector.transform(train_X)
        
        selector = VarianceThreshold()
        selector.fit(train_X)
        
        
        
        print("Classifiers will be fitted with {} out of {} features".format(len(feats), train_X.shape[1]))
        
        for name, m, fp in get_tree_models():
        
            with timer("Run " + name):
                
                model = AveragingModels(m, nfolds = 5, stratified = False)
                
                model.fit(train_X.loc[:, feats], train_y, **fp)
                pred = model.predict_proba(test_X.loc[:, feats])
                
                cv_score = model.auc_score_
                feat_importance = model.importances_
                
                if debug_size == 0:
                    submission = pp.submit_file(ids, pred, prefix_file_name = name, cv_score = cv_score)
                
                del model, pred, cv_score, feat_importance
                gc.collect()
            
    del test_X, train_X, train_y, submission
    del features_variance
    gc.collect()


#Full AUC score 0.769662
#[993]   train's auc: 0.86315    validation's auc: 0.768789
#[1053]  train's auc: 0.867744   validation's auc: 0.767411 (StandardScaler)
#[1241]  train's auc: 0.879679   validation's auc: 0.767784 (Robust Scaler)
#[1247]  train's auc: 0.879185   validation's auc: 0.768503 (MinMaxScaler)