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
from torch.utils import data
from utils import HOG, load_data, plot_sample_predictions

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

    batch_sizes = [1024, 2048]
    lrs = [0.001, 0.01]
    pix_per_cell_opts = [(8, 8), (4, 4), (16, 16)]
    hidden_layer_divs = [(2, 4), (3, 6), (4, 8)]
    orientation_opts = [16, 8, 4]
    epochs = 500

    grid_results = list()
    grid_counter = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_combos = (
        len(batch_sizes) * len(lrs) * len(pix_per_cell_opts) * len(hidden_layer_divs) * len(orientation_opts)
    )
    for pix_per_cell in pix_per_cell_opts:
        for orient in orientation_opts:

            ##### LOAD TRAINING DATA #####
            X_train, y_train = load_data("../../CW_Dataset", "train", mode="sklearn")

            hog = HOG(
                orientations=orient,
                pix_per_cell=pix_per_cell,
                cells_per_block=(1, 1),
                multichannel=True,
            )
            hog_arr = hog.fit_transform(X_train)

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

                        tensor_X_train = torch.tensor(hog_arr).float()
                        tensor_y_train = torch.tensor(y_train)

                        train_dataset = data.TensorDataset(tensor_X_train, tensor_y_train)
                        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                        ##### LOAD VALIDATION DATA #####
                        X_val, y_val = load_data("../../CW_Dataset", "val", mode="sklearn")

                        hog_arr_val = hog.fit_transform(X_val)
                        tensor_X_val = torch.tensor(hog_arr_val).float()
                        tensor_y_val = torch.tensor(y_val)
                        val_dataset = data.TensorDataset(tensor_X_val, tensor_y_val)
                        val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                        t0 = time.time()

                        hl_1 = hog_arr[0].shape[0] // hid_layer_div[0]
                        hl_2 = hog_arr[0].shape[0] // hid_layer_div[1]

                        ######## Create the model
                        net = EmotionRecMLP(
                            hog_arr[0].shape[0],
                            hl_1,
                            hl_2,
                            len(set(y_train)),
                        )
                        net = net.float()
                        net.to(device)

                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.95)

                        #### BEGIN THE TRAINING
                        losses = list()
                        val_accuracy = list()
                        for epoch in range(epochs):

                            val_iter = itertools.cycle(val_dataloader)

                            running_loss = 0.0
                            epoch_val_accuracy_arr = list()
                            for batch_idx, data_batch in enumerate(train_dataloader, 0):
                                # get the inputs; data is a list of [inputs, labels]
                                inputs, labels = data_batch[0].to(device), data_batch[1].to(device)

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
                                    images, labels = images.to(device), labels.to(device)
                                    outputs = net(images)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total = labels.size(0)
                                    correct = (predicted == labels).sum().item()
                                    val_batch_accuracy = 100 * correct / total
                                    epoch_val_accuracy_arr.append(val_batch_accuracy)

                                if batch_idx % 5 == 0:
                                    logger.info(
                                        f"Epoch: {epoch + 1} Loss: {running_loss / ((batch_idx + 1) * batch_size): .5f} "
                                        f"Validation Accuracy: {val_batch_accuracy: .2f}%"
                                    )
                            epoch_loss = running_loss / len(train_dataloader)
                            epoch_accuracy = np.mean(epoch_val_accuracy_arr)
                            val_accuracy.append(epoch_accuracy)
                            losses.append(epoch_loss)

                            # plot accuracy and loss every ten epochs
                            if epoch % 25 == 0:
                                fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
                                ax.set_xlabel("Epoch")
                                ax.plot(losses, color="b")
                                ax.set_ylabel("Training Loss", color="b")
                                ax2 = ax.twinx()
                                ax2.set_ylabel("Validation Accuracy", color="orange")
                                ax2.plot(val_accuracy, color="orange")
                                ax2.grid()
                                fig.tight_layout()
                                plt.show()

                        elapsed = time.time() - t0

                        max_accuracy = max(val_accuracy)
                        max_acc_epoch = np.array(val_accuracy).argmax() + 1
                        logger.info(f"Max accuracy achieved: {max_accuracy: .2f}% at epoch {max_acc_epoch}")

                        model_path = "..\\..\\Models"
                        model_name = f"hog_mlp_{time.strftime('%Y-%m-%d %H-%S')}.pth"

                        model_file_name = model_path + "/" + model_name
                        torch.save(net.state_dict(), model_file_name)

                        # VALIDATION ON ALL VAL DATA AT ONCE
                        ### Reset the validation datsets for overall metrics
                        val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                        correct = 0
                        total = 0
                        # create tensor for all predictions
                        # calculate accuracy for entire validation set
                        all_preds = torch.tensor([]).to(device)
                        with torch.no_grad():  # Avoid backprop at test
                            for data_batch in val_dataloader:
                                images, labels = data_batch
                                images, labels = images.to(device), labels.to(device)
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                all_preds = torch.cat((all_preds, predicted), dim=0)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                        unique_labels = np.unique(np.concatenate([y_val, all_preds.cpu()], axis=0))
                        cm = metrics.confusion_matrix(
                            val_dataset.tensors[1],
                            all_preds.cpu(),
                            labels=unique_labels,
                        )
                        f = sns.heatmap(
                            cm,
                            annot=True,
                            xticklabels=list([v for k, v in LABELS.items() if k != -1]),
                            yticklabels=list([v for k, v in LABELS.items() if k != -1]),
                            fmt="g",
                        )
                        
                        plt.show()

                        accuracy = 100 * correct / total
                        results["batch_size"] = batch_size
                        results["learning_rate"] = lr
                        results["pix_per_cell"] = pix_per_cell
                        results["orientations"] = orient
                        results["hidden_layer_div"] = hid_layer_div
                        results["elapsed"] = elapsed
                        results["accuracy"] = accuracy
                        results["recall"] = metrics.recall_score(y_val, all_preds.cpu(), average="weighted")
                        results["precision"] = metrics.precision_score(
                            y_val, all_preds.cpu(), average="weighted"
                        )
                        results["f1_score"] = metrics.f1_score(y_val, all_preds.cpu(), average="weighted")
                        results["max_val_acc"] = max_accuracy
                        results["max_acc_epoch"] = max_acc_epoch
                        results['hl_1'] = hl_1
                        results['hl_2'] = hl_2
                        results['model_name'] = model_name


                        grid_counter += 1
                        grid_results.append(results)
                        logger.info(str(results))
                        logger.info(print(metrics.classification_report(y_val, all_preds.cpu())))

                        df = pd.DataFrame.from_records(grid_results)
                        df.to_csv(
                            f"../../Outputs/MLP_grid_{time.strftime('%Y-%m-%d %H-%S')}.csv",
                            index=False,
                        )
                        plot_sample_predictions(X_val, all_preds.cpu(), y_val, 10)

    #     net = EmotionRecMLP(hog_arr[0].shape[0], hog_arr[0].shape[0] // 2, 7)
    #     net.load_state_dict(torch.load(model_file_name))
