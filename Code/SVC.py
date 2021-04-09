import logging

import numpy as np
from sklearn import svm
from utils import SIFT, MiniKMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionRecSVC:
    def __init__(self, kernel, C, cluster_factor, batch_divisor):
        self.batch_divisor = batch_divisor
        self.kernel = kernel
        self.C = C
        self.cluster_factor = cluster_factor
        self.svc_model = None
        self.KMeans = None

    def fit_transform(self, X, y):
        X = SIFT().fit_transform(X)
        idx_not_empty = [i for i, x in enumerate(X) if x is not None]
        X = [X[i] for i in idx_not_empty]
        y = [y[i] for i in idx_not_empty]

        self.KMeans = MiniKMeans(cluster_factor=self.cluster_factor, batch_divisor=self.batch_divisor).fit(X, y)
        X = self.KMeans.transform(X)

        logger.info(f"Fitting SVC with {self.kernel} kernel and C={self.C}")
        self.svc_model = svm.SVC(kernel=self.kernel, C=self.C)
        self.svc_model.fit(X, y)
        return self

    def predict(self, X):

        X_SIFT = SIFT().fit(X).transform(X)
        idx_not_empty = [i for i, x in enumerate(X_SIFT) if x is not None]
        idx_empty = [i for i, x in enumerate(X_SIFT) if x is None]
        X_valid = [X_SIFT[i] for i in idx_not_empty]
        X_valid = self.KMeans.transform(X_valid)

        predicted = self.svc_model.predict(X_valid)
        return_arr = np.zeros(len(X_SIFT))
        return_arr[idx_empty] = -1
        return_arr[idx_not_empty] = predicted
        return return_arr