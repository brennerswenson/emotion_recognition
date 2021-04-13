import argparse
import logging
import os
import time
from pathlib import Path
from pprint import pformat
import torch
from mlp import EmotionRecMLP
from sklearn import metrics
from utils import (DLDevice, SummaryWriter, eval_model, get_avail_device,
                   get_pred_metrics, load_data, plot_confusion_matrix,
                   plot_history, plot_sample_predictions, send_to_device,
                   train_model)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
DATASET_DIR = str(PROJECT_DIR.joinpath("cw_dataset"))
MODELS_DIR = str(PROJECT_DIR.joinpath("models"))
OUTPUTS_DIR = str(PROJECT_DIR.joinpath("outputs"))
RUN_NAME = f"MLP_{time.strftime('%Y-%m-%d %H-%M')}"

def main(args):
    writer = SummaryWriter(log_dir=str(PROJECT_DIR) + f"/logs/{RUN_NAME}")
    hid_layer_div = (2, 4, 8) if args.hidden_layer_divisor == "Halves" else (3, 6, 9)
    optimizer = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD
    logger.info(f"Input args: {pformat(args)}")

    hog_dict = {
        "orientation": args.orientation,
        "pix_per_cell": (args.pix_per_cell, args.pix_per_cell),
    }

    train_dl = load_data(
        DATASET_DIR,
        "train",
        "hog",
        hog_dict=hog_dict,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        weighted_sampling=True,
    )
    val_dl = load_data(DATASET_DIR, "val", "hog", hog_dict=hog_dict, batch_size=args.batch_size)

    for i, batch in enumerate(train_dl):
        HOG_des_arr, tmp_labels = batch
        # get the size of the input feature
        input_size = len(HOG_des_arr[0])
        break

    hl_1 = input_size // hid_layer_div[0]
    hl_2 = input_size // hid_layer_div[1]
    hl_3 = input_size // hid_layer_div[2]

    device = get_avail_device()
    train_dld = DLDevice(train_dl, device)
    val_dld = DLDevice(val_dl, device)
    model = EmotionRecMLP(
        input_size,
        hl_1,
        hl_2,
        hl_3,
        len(train_dl.dataset.classes),
        dropout_rate=args.dropout_rate
    )
    # just for adding model to tensorboard
    data_iter = iter(train_dl)
    images, labels = data_iter.next()
    writer.add_graph(model, images)
    model = send_to_device(model, device)
    hist = train_model(
        args.epochs,
        args.learning_rate,
        model,
        train_dld,
        val_dld,
        weight_decay=args.weight_decay,
        optimizer=optimizer,
        writer=writer,

    )
    logger.info((eval_model(model, val_dld)))
    model_name = f"{RUN_NAME}.pth"
    model_file_name = MODELS_DIR + "/" + model_name
    torch.save(model.state_dict(), model_file_name)

    all_preds, y_val, X_val, metrics_dict = get_pred_metrics(model, val_dld, device)
    logger.info(metrics_dict)
    logger.info(print(metrics.classification_report(y_val.cpu(), all_preds.cpu())))
    unique_labels = [int(x) - 1 for x in val_dl.dataset.classes]
    plot_confusion_matrix(y_val.cpu(), all_preds.cpu(), unique_labels, model='HOG-MLP', no_preds=False, writer=writer)
    # load it in again to get the images instead of HOG feature descriptors
    X_val, y_val = load_data(
        DATASET_DIR,
        "val",
        "normal",
        hog_dict=hog_dict,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        weighted_sampling=True,
    )
    plot_sample_predictions(
        X_val,
        all_preds.cpu(),
        y_val,
        4, 5,
        "HOG-MLP",
        tensor=False,
        writer=writer
    )
    plot_history(hist, writer)
    writer.add_hparams(
        hparam_dict={
            "Pixels Per Cell": hog_dict["pix_per_cell"][0],
            "Orientation": args.orientation,
            "Kernel": None,
            "C": 0,
            "Cluster Factor": 0,
            "Batch Divisor": 0,
            "Dropout Rate": args.dropout_rate,
            "Batch Size": args.batch_size,
            "Max Learning Rate": args.learning_rate,
            "Epochs": args.epochs,
            "Optimizer": args.optimizer,
            "Weight Decay": args.weight_decay,
        },
        metric_dict=metrics_dict
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        help="Batch size for training"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=1,
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "-wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        help="Weight decay constant"
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Max learning rate used during training",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        default="adam",
        help="Type of optimizer to use during training",
        choices=["adam", "sgd"],
    )

    parser.add_argument("-or", "--orientation", default=8, type=int, help="Orientations for HOG")
    parser.add_argument(
        "-p", "--pix-per-cell", default=4, type=int, help="Pixels per cell for HOG"
    )
    parser.add_argument(
        "-hl",
        "--hidden-layer-divisor",
        default="Halves",
        choices=["Halves", "Thirds"],
        help="Ratios of hidden layer sizes",
    )
    parser.add_argument(
        "-d",
        "--dropout-rate",
        default=0.1,
        type=float,
        help="Dropout rate used on Linear layers of neural network.",
    )

    args = parser.parse_args()
    main(args)

