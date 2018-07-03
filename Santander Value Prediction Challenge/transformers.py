import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone, is_classifier
from sklearn.model_selection._split import check_cv
from scipy.stats import skew, kurtosis

def get_stat_funs():
    
    def get_percentiles():
        percentiles = []
        for q in np.arange(0.1, 1.0, 0.1):
            percentiles.append(lambda x: np.percentile(x, q=q))
        return percentiles

    stat_funs = []
    stats = [len, np.min, np.max, np.mean, np.std, skew, kurtosis] + get_percentiles()
    
    for stat in stats:
        stat_funs.append(
            lambda x: -1 if x[x != 0.0].size == 0 else stat(x[x != 0.0])
        )
        stat_funs.append(
            lambda x: -1 if np.unique(x[x != 0.0]).size == 0 else stat(np.unique(x[x != 0.0]))
        )
        stat_funs.append(
            lambda x: -1 if np.diff(x[x != 0.0]).size == 0 else stat(np.diff(x[x != 0.0]))
        )
        stat_funs.append(
            lambda x: -1 if np.diff(np.unique(x[x != 0.0])).size == 0 else stat(np.diff(np.unique(x[x != 0.0])))
        )
    
    return stat_funs

class UniqueTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, axis=1, accept_sparse=False):
        if axis == 0:
            raise NotImplementedError('axis is 0! Not implemented!')
        if accept_sparse:
            raise NotImplementedError('accept_sparse is True! Not implemented!')
        self.axis = axis
        self.accept_sparse = accept_sparse
        
    def fit(self, X, y=None):
        _, self.unique_indices_ = np.unique(X, axis=self.axis, return_index=True)
        return self
    
    def transform(self, X, y=None):
        return X[:, self.unique_indices_]

class StatsTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, stat_funs=None):
        self.stat_funs = stat_funs
    
    def _get_stats(self, row):
        stats = []
        for fun in self.stat_funs:
            stats.append(fun(row))
        return stats
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return np.apply_along_axis(self._get_stats, arr=X, axis=1)
    
class ClassifierTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator=None, n_classes=2, cv=3):
        self.estimator = estimator
        self.n_classes = n_classes
        self.cv = cv
    
    def _get_labels(self, y):
        y_labels = np.zeros(len(y))
        y_us = np.sort(np.unique(y))
        step = int(len(y_us) / self.n_classes)
        
        for i_class in range(self.n_classes):
            if i_class + 1 == self.n_classes:
                y_labels[y >= y_us[i_class * step]] = i_class
            else:
                y_labels[
                    np.logical_and(
                        y >= y_us[i_class * step],
                        y < y_us[(i_class + 1) * step]
                    )
                ] = i_class
        return y_labels
        
    def fit(self, X, y):
        y_labels = self._get_labels(y)
        cv = check_cv(self.cv, y_labels, classifier=is_classifier(self.estimator))
        self.estimators_ = []
        
        for train, _ in cv.split(X, y_labels):
            self.estimators_.append(
                clone(self.estimator).fit(X[train], y_labels[train])
            )
        return self
    
    def transform(self, X, y=None):
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        
        X_prob = np.zeros((X.shape[0], self.n_classes))
        X_pred = np.zeros(X.shape[0])
        
        for estimator, (_, test) in zip(self.estimators_, cv.split(X)):
            X_prob[test] = estimator.predict_proba(X[test])
            X_pred[test] = estimator.predict(X[test])
        return np.hstack([X_prob, np.array([X_pred]).T])