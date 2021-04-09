import logging
import os
import random

import cv2
import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from config import LABELS, PLOT_RC
from matplotlib import pyplot as plt
from skimage import color, feature, img_as_ubyte
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm

sns.set()

sns.set_style(rc=PLOT_RC)
sns.set_context("notebook", font_scale=0.9)
mpl.rcParams["figure.edgecolor"] = "black"
mpl.rcParams["axes.linewidth"] = 0.5

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
                transforms.GaussianBlur(7, (0.01, 0.75)),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomRotation(degrees=10),
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
                transforms.GaussianBlur(7, (0.01, 0.75)),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomRotation(degrees=10),
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
        num_workers = num_workers if num_workers <= 5 else 5
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
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
        return data.DataLoader(
            dataset,
            num_workers=num_workers,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
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


def get_accuracy(out, truth_labels):
    _, out = torch.max(out, dim=1)
    return torch.tensor(torch.sum(out == truth_labels).item() / len(out))


class BaseModel(nn.Module):
    def train_step(self, batch):
        batch_images, batch_labels = batch
        output = self(batch_images)
        batch_loss = F.cross_entropy(output, batch_labels)
        return batch_loss

    def valid_step(self, batch):
        batch_images, batch_labels = batch
        output = self(batch_images)
        batch_loss = F.cross_entropy(output, batch_labels)
        val_acc = get_accuracy(output, batch_labels)
        return {
            "validation_loss": batch_loss.detach(),
            "validation_acc": val_acc,
        }

    @staticmethod
    def valid_whole_epoch(batch_outputs):
        all_losses = [x["validation_loss"] for x in batch_outputs]
        total_epoch_loss = torch.stack(all_losses).mean()  # add all of the losses
        all_accs = [x["validation_acc"] for x in batch_outputs]
        average_batch_accuracy = torch.stack(
            all_accs
        ).mean()  # combine all of the accuracies
        return {
            "validation_loss": total_epoch_loss.item(),
            "validation_acc": average_batch_accuracy.item(),
        }

    @staticmethod
    def epoch_performance(ep, output):
        logger.info(
            f"Epoch: {ep}, train_loss: {output['training_loss']:.4f}, "
            f"val_loss: {output['validation_loss']:.4f}, "
            f"val_acc: {output['validation_acc']:.4f}"
        )


def get_avail_device():
    """Choose the GPU if available else choose the CPU."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def send_to_device(obj, chosen_device):
    """Move tensor to device if a valid tensor."""
    if isinstance(obj, (tuple, list)):
        # unpack the data if needed
        return [send_to_device(x, chosen_device) for x in obj]
    return obj.to(chosen_device, non_blocking=True)


class DLDevice:
    """Class to move batches to the device."""

    def __init__(self, data_loader, dev):
        self.data_loader = data_loader
        self.dev = dev

    def __iter__(self):
        """Iterator for each batch after sending batch to device."""
        for batch in self.data_loader:
            yield send_to_device(batch, self.dev)

    def __len__(self):
        """Explicitly state the number of batches."""
        return len(self.data_loader)


@torch.no_grad()
def eval_model(fit_model, validation_dl):
    fit_model.eval()
    result = [fit_model.valid_step(b) for b in validation_dl]
    return fit_model.valid_whole_epoch(result)


def plot_history(train_history):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    losses = [x["training_loss"] for x in train_history]
    validation_accuracy = [x["validation_acc"] for x in train_history]
    ax.set_xlabel("Epoch")
    ax.plot(losses, color="b")
    ax.set_ylabel("Training Loss", color="b")
    ax2 = ax.twinx()
    ax2.set_ylabel("Validation Accuracy (%)", color="orange")
    ax2.plot(validation_accuracy, color="orange")
    ax.set_title("Training Loss and Validation Accuracy for CNN")
    ax2.grid()
    fig.tight_layout()
    plt.show()


def get_learning_rate(opt):
    for pg in opt.param_groups:
        return pg["lr"]


def train_model(
    num_epochs,
    lr_max,
    nn_model,
    training_dl,
    validation_dl,
    weight_decay=0,
    gradient_clip=None,
    optimizer=torch.optim.SGD,
):
    train_history = list()
    opt = optimizer(nn_model.parameters(), lr_max, weight_decay=weight_decay)
    shd = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr_max, epochs=num_epochs, steps_per_epoch=len(training_dl)
    )
    logger.info(f"Beginning training for {num_epochs} epochs")
    for e in range(num_epochs):
        logger.info(f"Epoch {e} of {num_epochs}")
        nn_model.train()
        losses = list()
        learning_rates = list()
        for batch_idx, b in tqdm(enumerate(training_dl, 0), total=len(training_dl)):
            logger.info(f"Batch: {batch_idx + 1}/{len(training_dl)}")
            loss_item = nn_model.train_step(b)
            losses.append(loss_item)
            loss_item.backward()

            if gradient_clip:
                nn.utils.clip_grad_value_(nn_model.parameters(), gradient_clip)

            opt.step()
            opt.zero_grad()

            learning_rates.append(get_learning_rate(opt))
            shd.step()

        eval_res = eval_model(nn_model, validation_dl)
        eval_res["training_loss"] = torch.stack(losses).mean().item()
        eval_res['learning_rates'] = learning_rates
        nn_model.epoch_performance(e, eval_res)
        train_history.append(eval_res)
        plot_history(train_history)
    return train_history


def plot_confusion_matrix(y_val, all_preds, unique_labels, no_preds=False):
    cm = metrics.confusion_matrix(
        y_val.cpu(),
        all_preds.cpu(),
        labels=unique_labels,
    )
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)  # Sample figsize in inches
    ax.set_title("Validation Confusion Matrix")

    if no_preds:
        ticks = list([v for k, v in LABELS.items()])
    else:
        ticks = list([v for k, v in LABELS.items() if k != -1])

    f = sns.heatmap(
        cm,
        annot=True,
        xticklabels=ticks,
        yticklabels=ticks,
        fmt="g",
        ax=ax,
    )
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def get_pred_metrics(model, val_dl, device):
    correct = 0
    total = 0
    all_preds = torch.tensor([]).to(device)
    y_val = torch.tensor([]).to(device)
    X_val = torch.tensor([]).to(device)
    for b in val_dl:
        images, labels = b
        images, labels = images.float().to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds = torch.cat((all_preds, predicted), dim=0)
        y_val = torch.cat((y_val, labels), dim=0)
        X_val = torch.cat((X_val, images), dim=0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    metrics_dict = dict()
    accuracy = 100 * correct / total
    metrics_dict["accuracy"] = accuracy
    metrics_dict["recall"] = metrics.recall_score(
        y_val.cpu(), all_preds.cpu(), average="weighted"
    )
    metrics_dict["precision"] = metrics.precision_score(
        y_val.cpu(), all_preds.cpu(), average="weighted"
    )
    metrics_dict["f1_score"] = metrics.f1_score(
        y_val.cpu(), all_preds.cpu(), average="weighted"
    )
    return all_preds, y_val, X_val, metrics_dict
