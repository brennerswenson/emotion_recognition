import logging
import os
import random
from pprint import pformat

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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torchvision import datasets, transforms
from tqdm import tqdm
import warnings

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
    weight_samp=False,
    num_images=None,
):
    """
    Function to dynamically load and preprocess images depending on the data subset (train/test/valid)
    as well as the pre-processing methods (SIFT, HOG, CNN).

    Args:
        path (str): Directory containing image files.
        subset (str): String e.g. "train" or "test".
        method (str): Type of pre-processing strategy e.g. "hog", "sift", "svm".
        hog_dict (dict): Dictionary containing hyperparameters for HOG feature extraction.
        batch_size (int or None): Training batch size for DataLoader processing methods.
        shuffle (bool): Indicates whether the returned DataLodaer will shuffle every batch
        drop_last (bool): Indicates whether or not the last incomplete batch should be discarded or not
        weight_samp (bool): Indicates if random weighted sampling should be applied to DataLoader

    Returns: (DataLoader or tuple) Preprocessed dataset

    """
    if not os.path.exists(path):
        logger.info("Please download training/testing data to cw_dataset directory")
        return None

    # create a string identifier with the data subset and model type
    key = subset + "_" + method

    # define the image augmentation transforms
    augmentation_transforms = [
        transforms.ColorJitter(0.50, 0.50, 0.1, 0.1),
        transforms.GaussianBlur(3, (0.001, 1)),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(15, scale=(0.75, 1.25)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.50),
        transforms.ToPILImage(),
    ]

    transform_dict = {
        # the training transforms contain data augmentation
        "train_hog": transforms.Compose(
            [
                *augmentation_transforms,
                HOG(
                    orientations=hog_dict.get("orientation"),
                    pix_per_cell=hog_dict.get("pix_per_cell"),
                    cells_per_block=(3, 3),
                    multichannel=True,
                ),
            ]
        ),
        # validation/test transforms do not have data augmentation
        "val_hog": transforms.Compose(
            [
                HOG(
                    orientations=hog_dict.get("orientation"),
                    pix_per_cell=hog_dict.get("pix_per_cell"),
                    cells_per_block=(3, 3),
                    multichannel=True,
                ),
            ]
        ),
        "test_hog": transforms.Compose(
            [
                HOG(
                    orientations=hog_dict.get("orientation"),
                    pix_per_cell=hog_dict.get("pix_per_cell"),
                    cells_per_block=(3, 3),
                    multichannel=True,
                ),
            ]
        ),
        "train_sift": transforms.Compose(
            [
                *augmentation_transforms,
                SIFT(),
            ]
        ),
        "val_sift": transforms.Compose([SIFT()]),
        "test_sift": transforms.Compose([SIFT()]),
        "train_normal": transforms.Compose([transforms.ToTensor()]),
        "val_normal": transforms.Compose([transforms.ToTensor()]),
        "test_normal": transforms.Compose([transforms.ToTensor()]),
        "train_cnn": transforms.Compose(
            [
                *augmentation_transforms,
                transforms.ToTensor(),
            ]
        ),
        "val_cnn": transforms.Compose([transforms.ToTensor()]),
        "test_cnn": transforms.Compose([transforms.ToTensor()]),
    }

    dataset = MyDataLoader(os.path.join(path, subset), transform=transform_dict[key])

    # for the methods that don't use batches for training
    # iterate through the dataset one time using image augmentation
    # and return the images/labels in a tuple
    if method in ["normal", "sift"]:
        images = list()
        labels = list()
        file_paths = list()

        if num_images is not None:
            # if we are only pulling a random subset
            logger.info(f'Loading {num_images} images')
            random_img_idx = random.choices(range(len(dataset)), k=num_images)
            for i, (img, label, fp) in enumerate(dataset):
                if i in random_img_idx:
                    img = img.permute(1, 2, 0).numpy() if torch.is_tensor(img) else img
                    label = int(label)
                    images.append(img)
                    labels.append(label)
                    file_paths.append(fp)

                    if len(images) == num_images:
                        break
        else:
            # pulling all images
            for img, label, fp in dataset:
                img = img.permute(1, 2, 0).numpy() if torch.is_tensor(img) else img
                label = int(label)
                images.append(img)
                labels.append(label)
                file_paths.append(fp)

        logger.info(f"Successfully loaded {len(images)} images")
        return images, labels, file_paths
    # if CNN or MLP, then create and return a DataLoader
    else:
        num_workers = len(dataset.imgs) // batch_size // 2
        num_workers = num_workers if num_workers <= 6 else 6
        num_workers = num_workers if num_images is not None else 0
        if num_workers > 0:
            logger.info(
                f"{subset} DataLoader using {num_workers} workers for {len(dataset.imgs)} images"
            )
        sampler = None
        # to combat class imbalance, weighted sampling can be used
        if weight_samp:
            class_distributions = list(Counter(dataset.targets).values())
            weights = 1 / torch.Tensor(class_distributions).float()
            sample_weights = weights[dataset.targets]
            logger.info(
                f"Using stratified sampling for {subset} data with weights "
                f"{pformat(list(zip(list(weights.float().numpy()), dataset.classes)))}"
            )
            # create the sampler object with the specified weights for each target class
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
    """Class for creating SIFT feature descriptors."""

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

    def __call__(self, x):
        kp, des = self.sift.detectAndCompute(img_as_ubyte(x), None)
        return des


class MiniKMeans:
    """Class for clustering input data, e.g. SIFT descriptors."""

    def __init__(self, cluster_factor, batch_divisor):
        """

        Args:
            cluster_factor (int): Factor to multiply the unique number of classes by. Determines the
                    number of clusters k used in clustering process.
            batch_divisor (int): Factor to divide the number of samples by. This dynamically creates the
                    batch size passed to MiniBatchKMeans.
        """
        self.batch_divisor = batch_divisor
        self.cluster_factor = cluster_factor
        self.num_clusters = None
        self.batch_size = None
        self.model = None

    def fit(self, X, y):
        """
        Instantiate and fit a MiniBatchKMeans object and set self.model attribute. Returns the class instance.
        Args:
            X (np.array): Array containing data to be clustered.
            y (np.array): Array containing the class labels of data.

        Returns: Class instance

        """
        X = np.vstack(X)
        self.num_clusters = len(np.unique(y)) * self.cluster_factor
        self.batch_size = X.shape[0] // self.batch_divisor
        logger.info(f"Fitting KMeans with {self.num_clusters} clusters")
        self.model = MiniBatchKMeans(n_clusters=self.num_clusters, batch_size=self.batch_size).fit(
            X
        )
        return self

    def transform(self, X):
        """
        Transform the provided array using the already fit KMeans object. Returns array of histograms.
        Args:
            X (np.array): Array containing new data to be clustered.

        Returns: (np.array) Array of concatenated histograms with cluster predictions.

        """
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
    """Class for extracting HOG feature descriptors from input images."""

    def __init__(self, orientations, pix_per_cell, cells_per_block, multichannel):
        """

        Args:
            orientations (int): Number of orientations used.
            pix_per_cell (tuple of int): Pixels per cell used.
            cells_per_block (tuple of int): Cells per block used.
            multichannel (bool): Indicates if HOG object input will be 3 channels or 1.
        """
        self.orientations = orientations
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block
        self.multichannel = multichannel

    def fit(self, X):
        return self

    def transform(self, X):
        """
        Iterate through all images and extract HOG feature descriptors
        using the hyperparameters provided during instantiation.
        Args:
            X (np.array): Array of images.

        Returns (list): List of hog feature descriptors.

        """
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
        """Override call method for use in PyTorch transform pipelines."""
        HOG_des = feature.hog(
            x,
            orientations=self.orientations,
            pixels_per_cell=self.pix_per_cell,
            cells_per_block=self.cells_per_block,
            feature_vector=True,
            multichannel=True,
        )
        HOG_des = torch.tensor(HOG_des, dtype=torch.float)
        return HOG_des


def plot_sample_predictions(
    X_test,
    y_pred,
    y_true,
    num_rows,
    num_cols,
    model_type,
    tensor=False,
    writer=None,
    figsize=(16, 9),
    accuracy=None,
):
    """
    Plot sample predictions for a given dataset of images. Depending on the number
    of columns and rows passed, num_rows * num_cols images will be randomly selected
    from the X_test array and plotted along with the predictions and truth labels.
    Args:
        X_test (np.array): Array containing images to be plotted.
        y_pred (np.array): Array of predictions (e.g. 1, 2, 3)
        y_true (np.array): Truth data to compare predictions against.
        num_rows (int): Number of rows in output plot of example predictions.
        num_cols (int): Number of columns in output plot of example predictions.
        model_type (str): Model used in prediction, e.g. CNN, MLP, SVM.
        tensor (bool): Indicates if input data is in tensor form.
        writer (SummaryWriter): SummaryWriter object used for TensorBoard.
        figsize (tuple): Dimensions in (x, y) of output figure.
        accuracy (float): Overall accuracy of testing run.

    Returns: None

    """
    random_idx = random.sample(range(0, len(X_test)), num_rows * num_cols)

    X_test = np.array(X_test)[random_idx]
    y_pred = np.array(y_pred)[random_idx]
    y_true = np.array(y_true)[random_idx]

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, dpi=150
    )
    title_str = f"Sample predictions of {model_type}"
    title_str = title_str + f" | Accuracy:{accuracy: .2f}%" if accuracy else title_str
    fig.suptitle(
        title_str,
        fontsize=14,
    )
    ax = axes.ravel()

    for i in range(num_rows * num_cols):
        img = X_test[i].transpose(1, 2, 0) if tensor else X_test[i]
        ax[i].imshow(img)
        ax[i].set_title(f"Label: {LABELS[y_true[i]]} \n Prediction: {LABELS[y_pred[i]]}")
        ax[i].set_axis_off()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if writer:
        writer.add_figure("Sample Predictions", fig)
    else:
        plt.show()


def get_accuracy(out, truth_labels):
    _, out = torch.max(out, dim=1)
    return torch.tensor((torch.sum(out == truth_labels).item() / len(out)) * 100)


class BaseModel(nn.Module):
    """
    Base class for nn.Module with common training/validation
    functions implemented for use across all types of PyTorch models.
    """

    def train_step(self, batch):
        """
        Unpack batch, get predictions, calculate loss, return loss.
        Args:
            batch (tensor): PyTorch tensor containing images and labels.

        Returns: Cross entropy loss for entire batch.

        """
        batch_images, batch_labels = batch
        output = self(batch_images)
        batch_loss = F.cross_entropy(output, batch_labels)
        return batch_loss

    def valid_step(self, batch):
        """
        Unpack validation batch, get prediction, calculate loss, return
        dictionary containing loss and validation accuracy.
        Args:
            batch (tensor): PyTorch tensor containing images and labels.

        Returns: Dictionary of batch validation loss and accuracy.

        """
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
        """
        Iterate through results of all batches, and calculate epoch-level
        metrics for entire dataset.
        Args:
            batch_outputs: List of validation metrics for each batch.

        Returns: Dictionary of entire epoch validation and accuracy metrics.

        """
        all_losses = [x["validation_loss"] for x in batch_outputs]
        total_epoch_loss = torch.stack(all_losses).mean()  # add all of the losses
        all_accs = [x["validation_acc"] for x in batch_outputs]
        average_batch_accuracy = torch.stack(all_accs).mean()  # combine all of the accuracies
        return {
            "validation_loss": total_epoch_loss.item(),
            "validation_acc": average_batch_accuracy.item(),
        }

    @staticmethod
    def epoch_performance(ep, output):
        """Log the performance of the entire epoch using the output of valid_whole_epoch."""
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
        return [send_to_device(x, chosen_device) for x in obj if torch.is_tensor(x)]
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
    """Evaluate the model passed using a validation DataLoader."""
    fit_model.eval()
    result = [fit_model.valid_step(b) for b in validation_dl]
    return fit_model.valid_whole_epoch(result)


def plot_history(train_history, writer=None):
    """
    Plot the training loss and validation accuracy on the same figure
    for the entirety of the training run.
    Args:
        train_history (list of dict): List containing metrics for each epoch.
        writer (SummaryWriter or None): TensorBoard SummaryWriter object to save the figure.

    Returns: None

    """
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

    if writer:
        writer.add_figure("Validation Accuracy and Training Loss", fig)
    else:
        plt.show()


def get_learning_rate(opt):
    """Get the learning rate used in the epoch from the optimizer object."""
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
    writer=None,
):
    """
    Iterate through dataset num_epochs times, and iterate through the training_dl
    using the batch sizes defined in the DataLoader. Update weights and clear gradients
    for each batch, then evaluate the model's performance after each epoch using
    validation data. Write the data to a TensorBoard SummaryWriter if passed.
    Args:
        num_epochs (int): Number of epochs for training.
        lr_max (float): The learning rate passed to the learning rate scheduler.
        nn_model (nn.Module): PyTorch model for training.
        training_dl (DataLoader): PyTorch DataLoader containing training images and labels.
        validation_dl (DataLoader): PyTorch DataLoader containing validation images and labels.
        weight_decay (float): Value used for weight decay.
        gradient_clip (float): Value used for gradient clipping, if passed.
        optimizer (torch.optim.Optimizer): PyTorch Optimizer object. e.g. Adam, SGD.
        writer (SummaryWriter): TensorBoard SummaryWriter object to keep track of experiment.

    Returns: (list of dict) Array containing the training and validation metrics of training run.

    """
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
        for batch_idx, b in tqdm(
            enumerate(training_dl, 0),
            total=len(training_dl),
            bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
        ):
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
        eval_res["learning_rates"] = learning_rates
        nn_model.epoch_performance(e, eval_res)
        train_history.append(eval_res)
        if writer:
            writer.add_scalar("Loss/Training", eval_res["training_loss"], e)
            writer.add_scalar("Loss/Validation", eval_res["validation_loss"], e)
            writer.add_scalar(
                "Loss/Val-Train-Delta", eval_res["validation_loss"] - eval_res["training_loss"], e
            )
            _ = [writer.add_scalar("Learning Rate", lr) for lr in eval_res["learning_rates"]]
            writer.add_scalar("Validation Accuracy", eval_res["validation_acc"], e)
    return train_history


def plot_confusion_matrix(
    y_true,
    all_preds,
    unique_labels,
    model,
    writer=None,
    no_preds=False,
):
    """
    Create and plot a confusion matrix from provided truth data. Add to SummaryWriter
    if passed, otherwise display the plot.
    Args:
        y_true (np.array): Array with truth labels.
        all_preds (np.array): Array with predicted labels.
        unique_labels (list of int): List containing the unique possible classes.
        model (str): Type of model being analysed.
        writer (SummaryWriter): TensorBoard SummaryWriter object.
        no_preds (bool): If True, allows -1 as a class indicating no prediction was made.

    Returns: None

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = metrics.confusion_matrix(
            y_true,
            all_preds,
            labels=unique_labels,
        )
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.set_title(f"{model} Test Confusion Matrix")

    if no_preds:
        ticks = list([v for k, v in LABELS.items()])
    else:
        ticks = list([v for k, v in LABELS.items() if k != -1])

    f = sns.heatmap(cm, annot=True, xticklabels=ticks, yticklabels=ticks, fmt="g", ax=ax)
    fig.tight_layout()

    if writer:
        writer.add_figure("Confusion Matrix", fig)
    else:
        plt.show()


@torch.no_grad()
def get_pred_metrics(model, dataloader, device, num_images=None):
    """
    Iterate through the dataloader and make predictions for each batch.
    Returns predictions, the X samples, y samples, and image file paths associated
    with each image. File paths are useful for re-plotting images when feature descriptors
    have been used.
    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader containing images and labels.
        device (torch.device): Device to send the batches to for prediction.
        num_images (int or None): If supplied, only num_images will be predicted.

    Returns (tuple): all_preds, y_val, X_val, metrics_dict, img_file_paths

    """
    correct = 0
    total = 0
    all_preds = torch.tensor([]).to(device)
    y_val = torch.tensor([]).to(device)
    X_val = torch.tensor([]).to(device)
    img_file_paths = list()

    for i, b in enumerate(dataloader):
        try:
            images, labels, file_paths = b
        except ValueError:
            images, labels = b
            file_paths = None
        images, labels = images.float().to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds = torch.cat((all_preds, predicted), dim=0)
        y_val = torch.cat((y_val, labels), dim=0)
        X_val = torch.cat((X_val, images), dim=0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if file_paths:
            img_file_paths.extend(file_paths)

        if num_images is not None:
            if i + 1 >= num_images:
                break

    metrics_dict = dict()
    accuracy = 100 * correct / total
    metrics_dict["accuracy"] = accuracy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics_dict["recall"] = metrics.recall_score(
            y_val.cpu(), all_preds.cpu(), average="weighted"
        )
        metrics_dict["precision"] = metrics.precision_score(
            y_val.cpu(), all_preds.cpu(), average="weighted"
        )
        metrics_dict["f1_score"] = metrics.f1_score(
            y_val.cpu(), all_preds.cpu(), average="weighted"
        )
    return all_preds, y_val, X_val, metrics_dict, img_file_paths


class SummaryWriter(SummaryWriter):
    """taken from https://github.com/pytorch/pytorch/issues/32651#issuecomment-648340103"""

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        log_dir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=log_dir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


class MyDataLoader(datasets.ImageFolder):
    def __getitem__(self, index):
        """Extend original ImageFolder dataset to include file paths in return value."""
        orig_tup = super(MyDataLoader, self).__getitem__(index)
        img_path = self.imgs[index][0]
        tup_with_path = orig_tup + (img_path,)
        return tup_with_path
