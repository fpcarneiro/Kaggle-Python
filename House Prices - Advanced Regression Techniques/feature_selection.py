import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, NMF
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression, RFE
import matplotlib.pyplot as plt

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

def get_feature_importance(estimator, X, y):
    plt.style.use('ggplot')
    pipe_importance = Pipeline([('scaler', RobustScaler()),
                              ('estimator', estimator)])
    pipe_importance.fit(X, y)
    if(hasattr(estimator, "feature_importance")):
        importances = pd.DataFrame({"Feature Importance":pipe_importance.named_steps['estimator'].feature_importance()}, index=X.columns)
    elif(hasattr(estimator, "feature_importances_")):
        importances = pd.DataFrame({"Feature Importance":pipe_importance.named_steps['estimator'].feature_importances_}, index=X.columns)
    else:
        importances = pd.DataFrame({"Feature Importance":pipe_importance.named_steps['estimator'].coef_}, index=X.columns)
    importances.sort_values("Feature Importance", ascending=False, inplace = True)
    return (importances)

def plot_features_importances(df_importances, show_importance_zero = False):
    if show_importance_zero:
        df_importances.sort_values("Feature Importance").plot(kind="barh",figsize=(15,50))
    else:
        df_importances[df_importances["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,50))
    plt.xticks(rotation=90)
    plt.show()

### Removing features with low variance - VarianceThreshold
def list_features_low_variance(X, y, threshold = .98):
    variance_threshold = (threshold * (1 - threshold))
    pipe_variance = Pipeline([('scaler', RobustScaler()),
                              ('reduce_dim', VarianceThreshold(variance_threshold))])
    pipe_variance.fit(X, y)
    features_variance = list(X.loc[:, pipe_variance.named_steps['reduce_dim'].get_support()].columns)
    return (features_variance)

### Feature selection using SelectFromModel - Lasso(alpha = 0.000507)
def remove_features_from_model(estimator, scaler, X, y):
    pipe_select_from_model = Pipeline([('scaler', scaler),
                                       ('reduce_dim', SelectFromModel(estimator)),
                                       ])

    pipe_select_from_model.fit(X, y)
    features_select_from_model = list(X.loc[:, pipe_select_from_model.named_steps['reduce_dim'].get_support()].columns)
    return (features_select_from_model, pipe_select_from_model)

#train_X, train_y, test_X = pp.get_processed_datasets()
#
#features_variance = remove_features_low_variance(train_X, train_y)
#
#train_X = train_X[features_variance]
#test_X = test_X[features_variance]
#
#importances = get_feature_importance(Lasso(alpha=0.000507), train_X, train_y)
#plot_features_importances(importances, show_importance_zero = False)
#
#### Univariate feature selection - SelectKBest
#pipe_kbest = Pipeline([('scaler', RobustScaler()),
#    ('reduce_dim', SelectKBest(mutual_info_regression, k=102)),
#])
#
#pipe_kbest.fit(train_X, train_y)
#features_kbest = list(train_X.loc[:, pipe_kbest.named_steps['reduce_dim'].get_support()].columns)
#train_X_reduced = pipe_kbest.transform(train_X)
#test_X_reduced = pipe_kbest.transform(test_X)
#
#### Univariate feature selection - SelectPercentile
#pipe_percentile = Pipeline([('scaler', RobustScaler()),
#    ('reduce_dim', SelectPercentile(mutual_info_regression)),
#    ('lasso', Lasso(alpha = 0.000507))
#])
#
#hp = {'reduce_dim__percentile': np.linspace(10, 50, 10)}
#    
#hpg = cvl.HousePricesGridCV(pipe_percentile, hyperparameters = hp, n_folds = 3)
#hpg.fit(train_X, train_y)
#hpg.get_best_results()
#hpg.grid_search.cv_results_
#hpg.plot_scores()
#
#
#hyperparameters_lasso = {'alpha': np.linspace(0.00002, 0.0008, 30)}
#hpg = cvl.HousePricesGridCV(Lasso(), hyperparameters = hyperparameters_lasso, n_folds = 5)
#hpg.fit(train_X, train_y)
#hpg.grid_search.cv_results_
#hpg.get_best_results()
#hpg.plot_scores()
#
#
#pipe_percentile = Pipeline([('scaler', RobustScaler()),
#    ('reduce_dim', SelectPercentile(mutual_info_regression, percentile=90)),
#])
#
#pipe_percentile.fit(train_X, train_y)
#features_percentile = list(train_X.loc[:, pipe_percentile.named_steps['reduce_dim'].get_support()].columns)
#train_X_reduced = pipe_percentile.transform(train_X)
#test_X_reduced = pipe_percentile.transform(test_X)
#
#### Recursive feature elimination - RFE
#pipe_recursive_selector = Pipeline([('scaler', RobustScaler()),
#    ('reduce_dim', RFE(Lasso(alpha = 0.000507), 110, step=1))
#])
#
#pipe_recursive_selector.fit(train_X, train_y)
#features_recursive_selector = list(train_X.loc[:, pipe_recursive_selector.named_steps['reduce_dim'].get_support()].columns)
#train_X_reduced = pipe_recursive_selector.transform(train_X)
#test_X_reduced = pipe_recursive_selector.transform(test_X)
#
##########################################################################################################
#all_predictors = train_X.columns
#predictors = list(set(train_X.columns) - set(was_missing_columns))
#
#import matplotlib.pyplot as plt
#import seaborn as sns
#import warnings
#plt.style.use('ggplot')
#lasso = Lasso(alpha=0.000507)
#lasso.fit(train_X_reduced, train_y)
#FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=features_select_from_model)
#FI_lasso.sort_values("Feature Importance",ascending=False)
#FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
#plt.xticks(rotation=90)
#plt.show()
#
#
#
#
##########################################################################################################
#
#model_lasso = Lasso(alpha = 0.00044, random_state = 1)
#model_ridge = Ridge(alpha=10.0)
#model_svr = SVR(C = 15, epsilon = 0.009, gamma = 0.0004, kernel = 'rbf')
#model_ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3, max_iter = 10000)
#model_KRR = KernelRidge(alpha=0.5, kernel='polynomial', degree=2, coef0=2.5)
#model_byr = BayesianRidge()
#model_rforest = RandomForestRegressor(n_estimators = 210)
#
#model_lsvr = LinearSVR()
#model_sgd = SGDRegressor()
#model_extra = ExtraTreesRegressor()
#
#model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
#                             learning_rate=0.05, max_depth=4, 
#                             min_child_weight=1.7817, n_estimators=3000,
#                             reg_alpha=0.4640, reg_lambda=0.88,
#                             subsample=0.5213, silent=1,
#                             random_state =7, nthread = -1)
#
#model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                   max_depth=4, max_features='sqrt',
#                                   min_samples_leaf=15, min_samples_split=10, 
#                                   loss='huber', random_state =5)
#
#model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
#                              learning_rate=0.05, n_estimators=720,
#                              max_bin = 55, bagging_fraction = 0.8,
#                              bagging_freq = 5, feature_fraction = 0.2319,
#                              feature_fraction_seed=9, bagging_seed=9,
#                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
#
#model_lasso_lars = LassoLars(alpha = 0.000507)
#
##Linear Models
#models = []
#models.append(("lasso", model_lasso))
#models.append(("ridge", model_ridge))
#models.append(("svr", model_svr))
#models.append(("ENet", model_ENet))
#models.append(("KRR", model_KRR))
#models.append(("byr", model_byr))
#models.append(("rforest", model_rforest))
#models.append(("xgb", model_xgb))
#models.append(("GBoost", model_GBoost))
#models.append(("lgb", model_lgb))
#models.append(("lasso_lars", model_lasso_lars))
#models.append(("lsvr", model_lsvr))
#
##models.append(("sgd", model_sgd))
##models.append(("extra", model_extra))
#
#scores = []
#names = []
#for name, model in models:
#    names.append(name)
#    scores.append(np.sqrt(pp.score_model(model, train_X_reduced, train_y)).mean())
#tab = pd.DataFrame({ "Model" : names, "Score" : scores })
#tab = tab.sort_values(by=['Score'], ascending = True)
#print(tab)
#
#for name, model in models:
#    model.fit(train_X_reduced, train_y)
#    
#averaged_models = em.AveragingModels(models = [model_byr, model_ENet, model_KRR])
#
#score_avg = np.sqrt(pp.score_model(averaged_models, train_X_reduced, train_y))
#print(" Averaged base models score: {:.6f} ({:.6f})\n".format(score_avg.mean(), score_avg.std()))
#
#averaged_models.fit(train_X_reduced, train_y)
#predicted_prices_averaged = np.expm1(averaged_models.predict(test_X_reduced))
#print(predicted_prices_averaged)
#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices_averaged})
#my_submission.to_csv('submission_avg.csv', index=False)
#
#stacked_averaged_models = em.StackingAveragedModels(base_models = [model_byr, model_ENet, model_lasso],
#                                                 meta_model = model_KRR)
#
#score_stacked_averaged = np.sqrt(pp.score_model(stacked_averaged_models, train_X_reduced, train_y))
#print(" Stacked Averaged base models score: {:.6f} ({:.6f})\n".format(score_stacked_averaged.mean(), 
#      score_stacked_averaged.std()))
#
#stacked_averaged_models.fit(train_X_reduced, train_y)
#predicted_prices_stacked_averaged = np.expm1(stacked_averaged_models.predict(test_X_reduced))
#print(predicted_prices_stacked_averaged)
#
#model_xgb.fit(train_X_reduced, train_y)
#predicted_prices_xgboost = np.expm1(model_xgb.predict(test_X_reduced))
#print(predicted_prices_xgboost)
#
#model_lgb.fit(train_X_reduced, train_y)
#predicted_prices_lgb = np.expm1(model_lgb.predict(test_X_reduced))
#print(predicted_prices_lgb)
#
#predicted_prices = predicted_prices_stacked_averaged*0.7 + predicted_prices_xgboost*0.2 + predicted_prices_lgb*0.1
#
#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
#my_submission.to_csv('submission_ensemble.csv', index=False)
#
#
#
#
#
#
#
#
#
## VALIDATION CURVE
#from sklearn.model_selection import validation_curve
#
#param_range = np.logspace(-1, 3, 30)
#param_range = np.linspace(10, 100, 100)
#train_scores, test_scores = validation_curve(Ridge(), train_X_reduced, train_y, param_name="alpha", param_range=param_range, 
#                                              scoring="neg_mean_squared_error", cv=5)
#
#train_scores = np.sqrt(-(train_scores))
#test_scores = np.sqrt(-(test_scores))
#
#train_scores_mean = np.mean(train_scores, axis=1)
#train_scores_std = np.std(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#test_scores_std = np.std(test_scores, axis=1)
#
#plt.title("Validation Curve with Ridge")
#plt.xlabel("Alpha")
#plt.ylabel("Score")
#plt.ylim(0.0, 0.2)
#
#lw = 2
#plt.semilogx(param_range, train_scores_mean, label="Training score",
#             color="darkorange", lw=lw)
#plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                 train_scores_mean + train_scores_std, alpha=0.2,
#                 color="darkorange", lw=lw)
#plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#             color="navy", lw=lw)
#plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                 test_scores_mean + test_scores_std, alpha=0.2,
#                 color="navy", lw=lw)
#plt.legend(loc="best")
#plt.show()
#
#
#
## LEARNING CURVE
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#from sklearn.datasets import load_digits
#from sklearn.model_selection import learning_curve
#from sklearn.model_selection import ShuffleSplit,KFold
#
#
#def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring="neg_mean_squared_error",
#                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 20)):
#    """
#    Generate a simple plot of the test and training learning curve.
#
#    Parameters
#    ----------
#    estimator : object type that implements the "fit" and "predict" methods
#        An object of that type which is cloned for each validation.
#
#    title : string
#        Title for the chart.
#
#    X : array-like, shape (n_samples, n_features)
#        Training vector, where n_samples is the number of samples and
#        n_features is the number of features.
#
#    y : array-like, shape (n_samples) or (n_samples, n_features), optional
#        Target relative to X for classification or regression;
#        None for unsupervised learning.
#
#    ylim : tuple, shape (ymin, ymax), optional
#        Defines minimum and maximum yvalues plotted.
#
#    cv : int, cross-validation generator or an iterable, optional
#        Determines the cross-validation splitting strategy.
#        Possible inputs for cv are:
#          - None, to use the default 3-fold cross-validation,
#          - integer, to specify the number of folds.
#          - An object to be used as a cross-validation generator.
#          - An iterable yielding train/test splits.
#
#        For integer/None inputs, if ``y`` is binary or multiclass,
#        :class:`StratifiedKFold` used. If the estimator is not a classifier
#        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
#
#        Refer :ref:`User Guide <cross_validation>` for the various
#        cross-validators that can be used here.
#
#    n_jobs : integer, optional
#        Number of jobs to run in parallel (default 1).
#    """
#    plt.figure()
#    plt.title(title)
#    if ylim is not None:
#        plt.ylim(*ylim)
#    plt.xlabel("Training examples")
#    plt.ylabel("Score")
#    train_sizes, train_scores, test_scores = learning_curve(
#        estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#    
#    
#    #train_scores = np.sqrt(-(train_scores))
#    #test_scores = np.sqrt(-(test_scores))
#
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    test_scores_std = np.std(test_scores, axis=1)
#
#    plt.grid()
#
#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std, alpha=0.1,
#                     color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
#    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#             label="Training score")
#    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#             label="Cross-validation score")
#
#    plt.legend(loc="best")
#    return plt
#
#
#title = "Learning Curves (Lasso)"
## Cross validation with 100 iterations to get smoother mean test and train
## score curves, each time with 20% data randomly selected as a validation set.
##cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
#cv = KFold(5, shuffle=True, random_state=42).get_n_splits(train_X_reduced)
#
#estimator = Lasso(alpha = 0.000507)
#plot_learning_curve(estimator, title, train_X_reduced, train_y, ylim=(0, 0.2), cv=cv, n_jobs=4)
#
#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
## SVC is more expensive so we do a lower number of CV iterations:
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#estimator = SVC(gamma=0.001)
#plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
#
#plt.show()