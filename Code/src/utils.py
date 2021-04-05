import os
import logging

import cv2
import numpy as np
from skimage import io, img_as_ubyte, color
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from matplotlib import pyplot as plt

from config import LABELS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path, subset):
    if not os.path.exists(path):
        logger.info(
            "Please download training/testing data to CW_Dataset directory"
        )
        return None
    else:
        images = list()
        labels = list()

    file_list = np.genfromtxt(
        path + f"/labels/list_label_{subset}.txt", dtype=str
    )
    logger.info(f"Loading {len(file_list)} images")
    for f_name, label in file_list:
        img = io.imread(
            os.path.join(
                path, subset, f_name.split(".jpg")[0] + "_aligned.jpg"
            )
        )
        if img is not None:
            images.append(img)
            labels.append(int(label))
        else:
            logger.info(f"Error loading image at {f_name}")
    logger.info(f"Successfully loaded {len(images)} images")
    return images, labels


class SIFT:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def fit(self, X):
        return self

    def transform(self, X):
        des_arr = list()
        logger.info(f"Beginning SIFT transformations for {len(X)} images")
        for i in tqdm(range(len(X))):
            img = img_as_ubyte(color.rgb2gray(X[i]))
            kp, des = self.sift.detectAndCompute(img, None)
            des_arr.append(des)
        return des_arr

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MiniKMeans:
    def __init__(self, cluster_factor, batch_divisor):
        self.batch_divisor = batch_divisor
        self.cluster_factor = cluster_factor
        self.num_clusters = None
        self.batch_size = None
        self.model = None

    def fit(self, X, y):
        X = np.vstack(X)
        self.num_clusters = len(np.unique(y)) * self.cluster_factor
        self.batch_size = X.shape[0] // self.batch_divisor
        logger.info(f"Fitting KMeans with {self.num_clusters} clusters")
        self.model = MiniBatchKMeans(
            n_clusters=self.num_clusters, batch_size=self.batch_size
        ).fit(X)
        return self

    def transform(self, X):
        hist_list = list()
        logger.info(f"Beginning clustering process for {len(X)} images")
        for des in tqdm(X):
            hist = np.zeros(self.num_clusters)
            idx = self.model.predict(des)
            for j in idx:
                hist[j] = hist[j] + (1 / len(des))

            hist_list.append(hist)
        return np.vstack(hist_list)


def plot_sample_predictions(X_test, y_pred, y_true, n_show):

    fig, axes = plt.subplots(
        2, 5, figsize=(14, 7), sharex=True, sharey=True, dpi=100
    )
    ax = axes.ravel()

    for i in range(n_show):
        ax[i].imshow(X_test[i])
        ax[i].set_title(
            f"Label: {LABELS[y_true[i]]} \n Prediction: {LABELS[y_pred[i]]}"
        )
        ax[i].set_axis_off()

    fig.tight_layout()
    plt.show()
