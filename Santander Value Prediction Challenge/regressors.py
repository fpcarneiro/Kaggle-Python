import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection._split import check_cv

class XGBRegressorCV(BaseEstimator, RegressorMixin):
    
    def __init__(self, xgb_params=None, fit_params=None, cv=3):
        self.xgb_params = xgb_params
        self.fit_params = fit_params
        self.cv = cv
    
    @property
    def feature_importances_(self):
        feature_importances = []
        for estimator in self.estimators_:
            feature_importances.append(
                estimator.feature_importances_
            )
        return np.mean(feature_importances, axis=0)
    
    @property
    def evals_result_(self):
        evals_result = []
        for estimator in self.estimators_:
            evals_result.append(
                estimator.evals_result_
            )
        return np.array(evals_result)
    
    @property
    def best_scores_(self):
        best_scores = []
        for estimator in self.estimators_:
            best_scores.append(
                estimator.best_score
            )
        return np.array(best_scores)
    
    @property
    def cv_scores_(self):
        return self.best_scores_ 
    
    @property
    def cv_score_(self):
        return np.mean(self.best_scores_)
    
    @property
    def best_iterations_(self):
        best_iterations = []
        for estimator in self.estimators_:
            best_iterations.append(
                estimator.best_iteration
            )
        return np.array(best_iterations)
    
    @property
    def best_iteration_(self):
        return np.round(np.mean(self.best_iterations_))
    
    def fit(self, X, y, **fit_params):
        cv = check_cv(self.cv, y, classifier=False)
        self.estimators_ = []
        
        for train, valid in cv.split(X, y):
            self.estimators_.append(
                xgb.XGBRegressor(**self.xgb_params).fit(
                    X[train], y[train],
                    eval_set=[(X[valid], y[valid])],
                    **self.fit_params
                )
            )

        return self
    
    def predict(self, X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis=0)