import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error

def score_sq(y, y_pred):
    return(np.sqrt(mean_squared_error(y, y_pred)))

class HousePricesGridCV():
    def __init__(self, model, hyperparameters, n_folds = 5, seed = None):
        self.model = model
        self.hyperparameter = hyperparameters
        self.folds = n_folds
        kf = KFold(self.folds, shuffle=True, random_state = seed)
        self.grid_search = GridSearchCV(self.model, self.hyperparameter,
                                   cv=kf, 
                                   scoring=make_scorer(score_sq, greater_is_better = False))
    
    def fit(self, X, y):
        self.grid_search.fit(X, y)
        self.grid_search.cv_results_['mean_test_score'] = -(self.grid_search.cv_results_['mean_test_score'])
        self.grid_search.cv_results_['mean_train_score'] = -(self.grid_search.cv_results_['mean_train_score'])
        #self.grid_search.best_score_ = -(self.grid_search.best_score_)
        
    def get_best_results(self):
        results = self.grid_search.cv_results_
        print(self.grid_search.best_params_, self.grid_search.best_score_)
        print(pd.DataFrame(results)[['params','mean_test_score', 'std_test_score']])
    
    def plot_scores(self):
        results = self.grid_search.cv_results_
        scorer = "score"
        color = 'g'
        # Get the regular numpy array from the MaskedArray
        for parameter, param_range in dict.items(self.hyperparameter):
            
            plt.figure(figsize=(10, 5))
            plt.title("GridSearchCV Evaluating", fontsize=15)
            X_axis = np.array(results['param_' + parameter].data, dtype=float)
            
            ax = plt.axes()
            #ax.set_xlim(min(X_axis), max(X_axis))
            ax.set_ylim(0.0, 0.20)
            
            for sample, style in (('train', '--'), ('test', '-')):
                sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
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

        
hyperparameters_ridge = {'alpha': np.linspace(7.5, 8.5, 25)}
hyperparameters_lasso = {'alpha': np.linspace(0.0001, 0.0009, 10)}
hyperparameters_svr = {'model__C':[1.366, 1.466], 'model__kernel':["rbf"], 
                       "model__gamma":[0.0003, 0.0004], 
                       "model__epsilon":[0.008, 0.009]}

hyperparameters_svr = {'C': np.linspace(1.0, 10, 10), 'epsilon' : np.linspace(0.005, 0.05, 10)}

hyperparameters_krr = {'model__alpha':[0.5, 0.6], 'model__kernel':["polynomial"], 'model__degree':[2], 'model__coef0':[2, 2.5]}
hyperparameters_ENet = {'model__alpha':[0.0008, 0.004, 0.0005],'model__l1_ratio':[0.08, 0.1, 0.9],'model__max_iter':[10000]}

hyperparameters_rforest = {"model__n_estimators" : np.arange(230, 250, 5), "model__max_depth": np.arange(10, 16, 2),
                           "model__bootstrap": [True, False]}
hyperparameters_xgb = {"model__booster" : ["gbtree", "gblinear"]}


hyperparameters_rforest = {"model__n_estimators" : [10], "model__max_depth": np.arange(1, 20, 2)}

hyper_models = []
#hyper_models.append((Lasso(), hyperparameters_lasso))
#hyper_models.append((Ridge(), hyperparameters_ridge))
#hyper_models.append((SVR(kernel = "rbf", gamma = 0.0004), hyperparameters_svr))
#hyper_models.append((KernelRidge(), hyperparameters_krr))
#hyper_models.append((ElasticNet(), hyperparameters_ENet))
#hyper_models.append((RandomForestRegressor(), hyperparameters_rforest))

#for hm, hp in hyper_models:
#    hpg = HousePricesGrid(hm, hyperparameters = hp, random = False, random_iter = 20, n_folds = 10)
#    hpg.fit(train_X, train_y)
#    hpg.get_best_results()
#    hpg.plot_scores()