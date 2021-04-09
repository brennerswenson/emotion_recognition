import itertools
import logging
import time

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from CNN import EmotionRecCNN
from config import LABELS
from matplotlib import pyplot as plt
from sklearn import metrics
from torch import nn
from tqdm import tqdm

from utils import load_data, plot_sample_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set()

rc = {
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "font.family": ["serif"],
    "grid.color": "gainsboro",
    "grid.linestyle": "-",
    "patch.edgecolor": "none",
}
sns.set_style(rc=rc)
sns.set_context("notebook", font_scale=0.9)
mpl.rcParams["figure.edgecolor"] = "black"
mpl.rcParams["axes.linewidth"] = 0.5

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    epochs = 5
    lr = 0.001

    train_dl = load_data(
        "../CW_Dataset",
        "train",
        "cnn",
        hog_dict=dict(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        weighted_sampling=True,
    )
    num_samples = len(train_dl.dataset.imgs)
    val_dl = load_data(
        "../CW_Dataset", "val", "cnn", hog_dict=dict(), batch_size=batch_size
    )
    t0 = time.time()

    net = EmotionRecCNN(output_size=num_samples)
    # net = torch.nn.DataParallel(net, device_ids=[0])
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.98)

    # BEGIN TRAINING
    losses = list()
    val_accuracy = list()

    for epoch in range(epochs):
        val_iter = itertools.cycle(val_dl)

        running_epoch_loss = 0.0
        epoch_val_accuracy_arr = list()

        for batch_idx, data_batch in tqdm(enumerate(train_dl, 0), total=len(train_dl)):
            inputs, labels = data_batch[0].float().to(device), data_batch[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_epoch_loss += loss.item() * inputs.size(0)

            # Validation batch accuracy
            with torch.no_grad():  # Avoid backprop at test
                images, labels = next(val_iter)
                images, labels = images.float().to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                val_batch_accuracy = 100 * correct / total
                epoch_val_accuracy_arr.append(val_batch_accuracy)

            if batch_idx % 1 == 0:
                logger.info(
                    f"Epoch: {epoch + 1}/{epochs} Batch: {batch_idx + 1}/{len(train_dl)} "
                    f"Loss: {running_epoch_loss / ((batch_idx + 1) * batch_size): .5f} "
                    f"Validation Accuracy: {val_batch_accuracy: .2f}%"
                )
        epoch_loss = running_epoch_loss / (len(train_dl) * batch_size)
        epoch_accuracy = np.mean(epoch_val_accuracy_arr)
        val_accuracy.append(epoch_accuracy)
        losses.append(epoch_loss)
        logger.info(
            f"Epoch: {epoch + 1}/{epochs} total accuracy: {epoch_accuracy: .2f}%"
        )

        # plot accuracy and loss every ten epochs
        if epoch % 1 == 0:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
            ax.set_xlabel("Epoch")
            ax.plot(losses, color="b")
            ax.set_ylabel("Training Loss", color="b")
            ax2 = ax.twinx()
            ax2.set_ylabel("Validation Accuracy (%)", color="orange")
            ax2.plot(val_accuracy, color="orange")
            ax.set_title("Training Loss and Validation Accuracy for CNN")
            ax2.grid()
            fig.tight_layout()
            plt.show()

    elapsed = time.time() - t0

    max_accuracy = max(val_accuracy)
    max_acc_epoch = np.array(val_accuracy).argmax() + 1
    logger.info(f"Max accuracy achieved: {max_accuracy: .2f}% at epoch {max_acc_epoch}")

    model_path = "../Models"
    model_name = f"CNN_{time.strftime('%Y-%m-%d %H-%S')}.pth"

    model_file_name = model_path + "/" + model_name
    torch.save(net.state_dict(), model_file_name)

    # VALIDATION ON ALL VAL DATA AT ONCE
    ### Reset the validation datsets for overall metrics
    val_dl = load_data(
        "../CW_Dataset",
        "val",
        "cnn",
        hog_dict=dict(),
        batch_size=batch_size,
        shuffle=False,
    )

    correct = 0
    total = 0
    # create tensors for all prediction data for overall stats
    # calculate accuracy for entire validation set
    all_preds = torch.tensor([]).to(device)
    y_val = torch.tensor([]).to(device)
    X_val = torch.tensor([]).to(device)
    with torch.no_grad():  # Avoid backprop at test
        for data_batch in val_dl:
            images, labels = data_batch
            images, labels = images.float().to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds = torch.cat((all_preds, predicted), dim=0)
            y_val = torch.cat((y_val, labels), dim=0)
            X_val = torch.cat((X_val, images), dim=0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    unique_labels = [int(x) - 1 for x in val_dl.dataset.classes]
    cm = metrics.confusion_matrix(
        y_val.cpu(),
        all_preds.cpu(),
        labels=unique_labels,
    )
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)  # Sample figsize in inches
    ax.set_title("Validation Confusion Matrix")
    f = sns.heatmap(
        cm,
        annot=True,
        xticklabels=list([v for k, v in LABELS.items() if k != -1]),
        yticklabels=list([v for k, v in LABELS.items() if k != -1]),
        fmt="g",
        ax=ax,
    )
    fig.tight_layout()

    plt.show()
    results = dict()
    accuracy = 100 * correct / total
    results["batch_size"] = batch_size
    results["learning_rate"] = lr
    results["elapsed"] = elapsed
    results["accuracy"] = accuracy
    results["recall"] = metrics.recall_score(
        y_val.cpu(), all_preds.cpu(), average="weighted"
    )
    results["precision"] = metrics.precision_score(
        y_val.cpu(), all_preds.cpu(), average="weighted"
    )
    results["f1_score"] = metrics.f1_score(
        y_val.cpu(), all_preds.cpu(), average="weighted"
    )
    results["max_val_acc"] = max_accuracy
    results["max_acc_epoch"] = max_acc_epoch
    results["model_name"] = model_name
    logger.info(str(results))
    logger.info(print(metrics.classification_report(y_val.cpu(), all_preds.cpu())))

    df = pd.DataFrame.from_records([results])
    df.to_csv(
        f"../Outputs/CNN_grid_{time.strftime('%Y-%m-%d %H-%S')}.csv",
        index=False,
    )
    X_val, y_val = load_data(
        "../CW_Dataset",
        "val",
        "cnn",
        hog_dict=dict(),
        batch_size=batch_size,
        shuffle=False,
    )
    plot_sample_predictions(X_val, all_preds.cpu(), y_val, 4, 5, "CNN")
