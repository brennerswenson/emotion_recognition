import itertools
import logging
import time

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from config import LABELS
from matplotlib import pyplot as plt
from MLP import EmotionRecMLP
from sklearn import metrics
from torch import nn
from utils import load_data, plot_sample_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from tqdm import tqdm

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

    batch_sizes = [512, 1024]
    lrs = [0.001]
    pix_per_cell_opts = [
        (4, 4),
        (3, 3),
    ]
    hidden_layer_divs = [
        (2, 4, 8),
        (3, 6, 9),
    ]
    orientation_opts = [8, 4]
    epochs = 100

    grid_results = list()
    grid_counter = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_combos = (
        len(batch_sizes) * len(lrs) * len(pix_per_cell_opts) * len(hidden_layer_divs) * len(orientation_opts)
    )
    for pix_per_cell in pix_per_cell_opts:
        for orient in orientation_opts:
            for batch_size in batch_sizes:
                for lr in lrs:
                    for hid_layer_div in hidden_layer_divs:
                        logger.info(f"{grid_counter + 1} out of {total_combos}")
                        logger.info(f"Batch size: {batch_size}")
                        logger.info(f"Learning rate: {lr}")
                        logger.info(f"Pixels per cell: {pix_per_cell}")
                        logger.info(f"Hidden layer divisors: {hid_layer_div}")
                        logger.info(f"Orientations: {orient}")

                        results = dict()

                        logger.info(f"Training on {device}")

                        hog_dict = {"orientation": orient, "pix_per_cell": pix_per_cell}
                        train_dl = load_data(
                            "../CW_Dataset",
                            "train",
                            "hog",
                            hog_dict,
                            batch_size,
                            shuffle=False,
                            drop_last=False,
                            weighted_sampling=True,
                        )
                        num_samples = len(train_dl.dataset.imgs)
                        val_dl = load_data("../CW_Dataset", "val", "hog", hog_dict, batch_size)

                        t0 = time.time()

                        ## get size of first batch to create
                        # neural network architecture
                        for i, batch in enumerate(train_dl):
                            HOG_des_arr, tmp_labels = batch
                            # get the size of the input feature
                            input_size = len(HOG_des_arr[0])
                            break

                        hl_1 = input_size // hid_layer_div[0]
                        hl_2 = input_size // hid_layer_div[1]
                        hl_3 = input_size // hid_layer_div[2]

                        ######## Create the model
                        net = EmotionRecMLP(
                            input_size,
                            hl_1,
                            hl_2,
                            hl_3,
                            len(val_dl.dataset.classes),
                        )
                        net = net.float()
                        net = torch.nn.DataParallel(net, device_ids=[0])
                        net.to(device)

                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.98)

                        #### BEGIN THE TRAINING
                        losses = list()
                        val_accuracy = list()
                        for epoch in range(epochs):
                            val_iter = itertools.cycle(val_dl)

                            running_loss = 0.0
                            epoch_val_accuracy_arr = list()
                            for batch_idx, data_batch in tqdm(enumerate(train_dl, 0), total=len(train_dl), ):
                                # get the inputs; data is a list of [inputs, labels]
                                inputs, labels = data_batch[0].float().to(device), data_batch[1].to(device)
                                # zero the parameter gradients
                                optimizer.zero_grad()

                                # forward + backward + optimize
                                outputs = net(inputs)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()

                                running_loss += loss.item() * inputs.size(0)

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
                                        f"Loss: {running_loss / ((batch_idx + 1) * batch_size): .5f} "
                                        f"Validation Accuracy: {val_batch_accuracy: .2f}%"
                                    )
                            epoch_loss = running_loss / (len(train_dl) * batch_size)
                            epoch_accuracy = np.mean(epoch_val_accuracy_arr)
                            val_accuracy.append(epoch_accuracy)
                            losses.append(epoch_loss)
                            logger.info(f"Epoch: {epoch + 1}/{epochs} total accuracy: {epoch_accuracy: .2f}%")

                            # plot accuracy and loss every ten epochs
                            if epoch % 1 == 0:
                                fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
                                ax.set_xlabel("Epoch")
                                ax.plot(losses, color="b")
                                ax.set_ylabel("Training Loss", color="b")
                                ax2 = ax.twinx()
                                ax2.set_ylabel("Validation Accuracy (%)", color="orange")
                                ax2.plot(val_accuracy, color="orange")
                                ax.set_title('Training Loss and Validation Accuracy for HOG-MLP')
                                ax2.grid()
                                fig.tight_layout()
                                plt.show()

                        elapsed = time.time() - t0

                        max_accuracy = max(val_accuracy)
                        max_acc_epoch = np.array(val_accuracy).argmax() + 1
                        logger.info(f"Max accuracy achieved: {max_accuracy: .2f}% at epoch {max_acc_epoch}")

                        model_path = "../Models"
                        model_name = f"hog_mlp_{time.strftime('%Y-%m-%d %H-%S')}.pth"

                        model_file_name = model_path + "/" + model_name
                        torch.save(net.state_dict(), model_file_name)

                        # VALIDATION ON ALL VAL DATA AT ONCE
                        ### Reset the validation datsets for overall metrics
                        val_dl = load_data(
                            "../CW_Dataset", "val", "hog", hog_dict, batch_size, shuffle=False
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
                        ax.set_title('Validation Confusion Matrix')
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

                        accuracy = 100 * correct / total
                        results["batch_size"] = batch_size
                        results["learning_rate"] = lr
                        results["pix_per_cell"] = pix_per_cell
                        results["orientations"] = orient
                        results["hidden_layer_div"] = hid_layer_div
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
                        results["hl_1"] = hl_1
                        results["hl_2"] = hl_2
                        results["model_name"] = model_name

                        grid_counter += 1
                        grid_results.append(results)
                        logger.info(str(results))
                        logger.info(print(metrics.classification_report(y_val.cpu(), all_preds.cpu())))

                        df = pd.DataFrame.from_records(grid_results)
                        df.to_csv(
                            f"../../Outputs/MLP_grid_{time.strftime('%Y-%m-%d %H-%S')}.csv",
                            index=False,
                        )
                        X_val, y_val = load_data(
                            "../CW_Dataset", "val", "normal", hog_dict, batch_size, shuffle=False
                        )
                        plot_sample_predictions(X_val, all_preds.cpu(), y_val, 4, 5, 'HOG-MLP')

    #     net = EmotionRecMLP(hog_arr[0].shape[0], hog_arr[0].shape[0] // 2, 7)
    #     net.load_state_dict(torch.load(model_file_name))
