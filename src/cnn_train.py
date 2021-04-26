
import argparse
import logging
import os
import time
from pathlib import Path
from pprint import pformat

import torch
from cnn import EmotionRecCNN
from sklearn import metrics
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# filepaths change depending on environment, so these variables
# allow for dynamic paths between local/colab
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
DATASET_DIR = str(PROJECT_DIR.joinpath("cw_dataset"))
MODELS_DIR = str(PROJECT_DIR.joinpath("models"))
OUTPUTS_DIR = str(PROJECT_DIR.joinpath("outputs"))
RUN_NAME = f"CNN_{time.strftime('%Y-%m-%d %H-%M')}"


def main(args):
    writer = utils.SummaryWriter(log_dir=str(PROJECT_DIR) + f"/logs/{RUN_NAME}")
    optimizer = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD
    logger.info(f"Input args: {pformat(args)}")

    train_dl = utils.load_data(
        DATASET_DIR,
        "train",
        "cnn",
        hog_dict=dict(),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        weight_samp=True,
    )

    val_dl = utils.load_data(
        DATASET_DIR,
        "val",
        "cnn",
        hog_dict=dict(),
        batch_size=args.batch_size
    )

    num_classes = len(train_dl.dataset.classes)
    device = utils.get_avail_device()
    train_dld = utils.DLDevice(train_dl, device)
    val_dld = utils.DLDevice(val_dl, device)
    model = EmotionRecCNN(output_size=num_classes, dropout_rate=args.dropout_rate)

    # just for adding model to tensorboard
    data_iter = iter(train_dl)
    images, labels = data_iter.next()
    writer.add_graph(model, images)

    model = utils.send_to_device(model, device)
    hist = utils.train_model(
        args.epochs,
        args.learning_rate,
        model,
        train_dld,
        val_dld,
        weight_decay=args.weight_decay,
        optimizer=optimizer,
        writer=writer,
    )
    logger.info((utils.eval_model(model, val_dld)))
    model_name = f"{RUN_NAME}.pth"
    model_file_name = MODELS_DIR + "/" + model_name
    torch.save(model.state_dict(), model_file_name)

    all_preds, y_val, X_val, metrics_dict = utils.get_pred_metrics(model, val_dld, device)

    logger.info(metrics_dict)
    logger.info(print(metrics.classification_report(y_val.cpu(), all_preds.cpu())))

    unique_labels = [int(x) - 1 for x in val_dl.dataset.classes]
    utils.plot_confusion_matrix(y_val.cpu(), all_preds.cpu(), unique_labels, model='CNN', no_preds=False, writer=writer)
    utils.plot_sample_predictions(
        X_val.cpu(), all_preds.cpu(), y_val.cpu(), 4, 5, "CNN", tensor=True, writer=writer
    )
    utils.plot_history(hist, writer)

    writer.add_hparams(
        hparam_dict={
            "Pixels Per Cell": 0,
            "Orientation": 0,
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
        metric_dict=metrics_dict,
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
        default=25,
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
    parser.add_argument(
        "-d",
        "--dropout-rate",
        default=0.25,
        type=float,
        help="Dropout rate used on Linear layers of neural network.",
    )
    args = parser.parse_args()
    main(args)
