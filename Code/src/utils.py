import os
import logging
import random

import cv2
import numpy as np
from skimage import img_as_ubyte, color, feature
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from config import LABELS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path, subset, mode="sklearn"):
    if not os.path.exists(path):
        logger.info("Please download training/testing data to CW_Dataset directory")
        return None

    transform_dict = {
        # the training transforms contain data augmentation
        "train": transforms.Compose(
            [
                transforms.ColorJitter(0.15, 0.15, 0.15, 0.15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "val": transforms.Compose([transforms.ToTensor()]),
        "test": transforms.Compose([transforms.ToTensor()]),
    }

    dataset = datasets.ImageFolder(path + "/" + subset, transform=transform_dict[subset])

    if mode == "sklearn":
        images = list()
        labels = list()

        for img, label in dataset:
            img = img.permute(1, 2, 0).numpy()
            label = int(label)
            images.append(img)
            labels.append(label)
        logger.info(f"Successfully loaded {len(images)} images")
        return images, labels
    else:
        return data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)


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


class HOG:
    def __init__(self, orientations, pix_per_cell, cells_per_block, multichannel):
        self.orientations = orientations
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block
        self.multichannel = multichannel

    def fit(self, X):
        return self

    def transform(self, X):
        hog_arr = list()

        logger.info(f"Beginning HOG transformations for {len(X)} images")
        for i in tqdm(range(len(X))):
            img = img_as_ubyte(color.rgb2gray(X[i]))
            HOG_des = feature.hog(
                img,
                orientations=self.orientations,
                pixels_per_cell=self.pix_per_cell,
                cells_per_block=self.cells_per_block,
                feature_vector=True,
                multichannel=False
            )
            hog_arr.append(HOG_des)
        return hog_arr

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def plot_sample_predictions(X_test, y_pred, y_true, n_show):

    random_idx = random.sample(range(0, len(X_test)), n_show)

    X_test = np.array(X_test)[random_idx]
    y_pred = np.array(y_pred)[random_idx]
    y_true = np.array(y_true)[random_idx]

    fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True, dpi=100)
    ax = axes.ravel()

    for i in range(n_show):
        ax[i].imshow(X_test[i])
        ax[i].set_title(f"Label: {LABELS[y_true[i]]} \n Prediction: {LABELS[y_pred[i]]}")
        ax[i].set_axis_off()

    fig.tight_layout()
    plt.show()
