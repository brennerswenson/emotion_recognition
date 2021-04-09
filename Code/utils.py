import logging
import os
import random

import cv2
import numpy as np
import torch
from config import LABELS
from matplotlib import pyplot as plt
from skimage import color, feature, img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from collections import Counter


def load_data(
    path,
    subset,
    method,
    hog_dict,
    batch_size,
    shuffle=True,
    drop_last=False,
    weighted_sampling=False,
):
    if not os.path.exists(path):
        logger.info("Please download training/testing data to CW_Dataset directory")
        return None

    key = subset + "_" + method

    transform_dict = {
        # the training transforms contain data augmentation
        "train_hog": transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.10),
                transforms.ColorJitter(0.50, 0.50, 0.05, 0.05),
                transforms.GaussianBlur(7, (0.01, 1)),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(100, (0.85, 1)),
                HOG(
                    orientations=hog_dict.get("orientation"),
                    pix_per_cell=hog_dict.get("pix_per_cell"),
                    cells_per_block=(1, 1),
                    multichannel=True,
                ),
            ]
        ),
        "val_hog": transforms.Compose(
            [
                HOG(
                    orientations=hog_dict.get("orientation"),
                    pix_per_cell=hog_dict.get("pix_per_cell"),
                    cells_per_block=(1, 1),
                    multichannel=True,
                )
            ]
        ),
        "train_normal": transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.10),
                transforms.ColorJitter(0.50, 0.50, 0.05, 0.05),
                transforms.GaussianBlur(7, (0.01, 1)),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(100, (0.85, 1)),
                transforms.ToTensor(),
            ]
        ),
        "val_normal": transforms.Compose([transforms.ToTensor()]),
        "train_cnn": transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.10),
                transforms.ColorJitter(0.50, 0.50, 0.05, 0.05),
                transforms.GaussianBlur(7, (0.01, 1)),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(100, (0.85, 1)),
                transforms.ToTensor(),
            ]
        ),
        "val_cnn": transforms.Compose([transforms.ToTensor()]),
    }

    dataset = datasets.ImageFolder(path + "/" + subset, transform=transform_dict[key])

    if method == "normal":
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
        num_workers = len(dataset.imgs) // batch_size // 2
        num_workers = num_workers if num_workers <= 12 else 12
        logger.info(
            f"{subset} DataLoader using {num_workers} workers for {len(dataset.imgs)} images"
        )
        sampler = None
        if weighted_sampling:
            class_distributions = list(Counter(dataset.targets).values())
            weights = 1 / torch.Tensor(class_distributions).float()
            sample_weights = weights[dataset.targets]
            logger.info(
                f"Using stratified sampling for {subset} data with weights "
                f"{list(zip(list(weights.float().numpy()), dataset.classes))}"
            )
            sampler = data.sampler.WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )
        return data.DataLoader(
            dataset,
            num_workers=num_workers,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )


class SIFT:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def fit(self, X):
        return self

    def transform(self, X):
        des_arr = list()
        logger.info(f"Beginning SIFT transformations for {len(X)} images")
        for i in tqdm(range(len(X))):
            kp, des = self.sift.detectAndCompute(img_as_ubyte(X[i]), None)
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
            n_clusters=self.num_clusters,
            batch_size=self.batch_size
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
                multichannel=False,
            )
            hog_arr.append(HOG_des)
        return hog_arr

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def __call__(self, x):
        HOG_des = feature.hog(
            x,
            orientations=self.orientations,
            pixels_per_cell=self.pix_per_cell,
            cells_per_block=self.cells_per_block,
            feature_vector=True,
            multichannel=True,
        )
        return HOG_des


def plot_sample_predictions(X_test, y_pred, y_true, num_rows, num_cols, model_type):

    random_idx = random.sample(range(0, len(X_test)), num_rows * num_cols)

    X_test = np.array(X_test)[random_idx]
    y_pred = np.array(y_pred)[random_idx]
    y_true = np.array(y_true)[random_idx]

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(16, 9), sharex=True, sharey=True, dpi=150
    )
    fig.suptitle(
        f"Sample predictions of {model_type}",
        fontsize=14,
    )
    ax = axes.ravel()

    for i in range(num_rows * num_cols):
        ax[i].imshow(X_test[i])
        ax[i].set_title(f"Label: {LABELS[y_true[i]]} \n Prediction: {LABELS[y_pred[i]]}")
        ax[i].set_axis_off()

    fig.tight_layout()
    plt.show()
