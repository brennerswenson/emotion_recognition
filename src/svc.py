import logging

import numpy as np
from sklearn import svm
from utils import MiniKMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionRecSVC:
    """Support Vector Classifier for classifying human emotions."""
    def __init__(self, kernel, C, cluster_factor, batch_divisor):
        """

        Args:
            kernel (str): Type of kernel to use in SVC.
            C (float): Regularization parameter used in SVC.
            cluster_factor (int): Factor to multiply the unique number of classes by. Determines the
                    number of clusters k used in clustering process.
            batch_divisor (int): Factor to divide the number of samples by. This dynamically creates the
                    batch size passed to MiniBatchKMeans
        """
        self.batch_divisor = batch_divisor
        self.kernel = kernel
        self.C = C
        self.cluster_factor = cluster_factor
        self.svc_model = None
        self.KMeans = None

    def fit_transform(self, X, y):
        """
        Fit MiniKMeans object on the SIFT descriptors, then fit the support vector classifier using
        the clustered SIFT data.
        Args:
            X (np.array): Array of SIFT feature descriptors for each image. Can be None.
            y: (np.array): Array of labels for each SIFT feature descriptor.

        Returns: Class instance.
        """
        idx_not_empty = [i for i, x in enumerate(X) if x is not None]
        X = [X[i] for i in idx_not_empty]
        y = [y[i] for i in idx_not_empty]

        self.KMeans = MiniKMeans(
            cluster_factor=self.cluster_factor,
            batch_divisor=self.batch_divisor
        ).fit(X, y)
        X = self.KMeans.transform(X)
        logger.info(f"Fitting SVC with {self.kernel} kernel and C={self.C}")
        self.svc_model = svm.SVC(kernel=self.kernel, C=self.C)
        self.svc_model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict on unseen SIFT descriptors. Remove and classify the null SIFT descriptors as -1 class.
        Returns an array of predictions.
        Args:
            X (np.array): Array of SIFT feature descriptors for each image. Can be None.

        Returns: (list) Array of class predictions.

        """
        idx_not_empty = [i for i, x in enumerate(X) if x is not None]
        idx_empty = [i for i, x in enumerate(X) if x is None]
        X_valid = [X[i] for i in idx_not_empty]
        X_valid = self.KMeans.transform(X_valid)

        predicted = self.svc_model.predict(X_valid)
        return_arr = np.zeros(len(X))
        return_arr[idx_empty] = -1
        return_arr[idx_not_empty] = predicted
        return return_arr
