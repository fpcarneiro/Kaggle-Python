from __future__ import division

from collections import Counter

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from statsmodels.distributions import ECDF

class LikelihoodEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, seed=0, alpha=0, leave_one_out=False, noise=0):
        self.alpha = alpha
        self.noise = noise
        self.seed = seed
        self.leave_one_out = leave_one_out
        self.nclass = None
        self.estimators = []

    def fit(self, x, y):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        if not is_numpy(x):
            x = np.array(x)

        self.nclass = np.unique(y).shape[0]

        for i in range(ncols):
            self.estimators.append(LikelihoodEstimator(**self.get_params()).fit(x[:, i], y))
        return self

    @staticmethod
    def owen_zhang(x_train, y_train, x_test, seed=0, alpha=0, noise=0.01):
        """
        Owen Zhang's leave-one-out + noise likelihood encoding
        "Winning data science competitions"
        http://de.slideshare.net/ShangxuanZhang/winning-data-science-competitions-presented-by-owen-zhang
        """
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)
        ncols = x_train.shape[1]
        nclass = np.unique(y_train).shape[0]
        if not is_numpy(x_train):
            x_train = np.array(x_train)
            x_test = np.array(x_test)

        xx_train = None
        xx_test = None

        for i in range(ncols):
            le_train = LikelihoodEstimator(noise=noise, alpha=alpha, leave_one_out=True, seed=seed). \
                fit(x_train[:, i], y_train)
            le_test = LikelihoodEstimator(noise=0, alpha=alpha, leave_one_out=False, seed=seed). \
                fit(x_train[:, i], y_train)
            lh_train = le_train.x_likelihoods.copy()
            lh_test = le_test.predict_proba(x_test[:, i])

            if nclass <= 2:
                lh_train = lh_train.T[1].reshape(-1, 1)
                lh_test = lh_test.T[1].reshape(-1, 1)

            xx_train = np.hstack((lh_train,)) if xx_train is None else np.hstack((xx_train, lh_train))
            xx_test = np.hstack((lh_test,)) if xx_test is None else np.hstack((xx_test, lh_test))

        return xx_train, xx_test

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        if not is_numpy(x):
            x = np.array(x)

        likelihoods = None

        for i in range(ncols):
            lh = self.estimators[i].predict(x[:, i], noise=True).reshape(-1, 1)
            # lh = self.estimators[i].predict_proba(x[:, i])
            # if self.nclass <= 2:
            #     lh = lh.T[1].reshape(-1, 1)
            likelihoods = np.hstack((lh,)) if likelihoods is None else np.hstack((likelihoods, lh))
        return likelihoods