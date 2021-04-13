import argparse
import logging
import os
import joblib
import time
import warnings
from pathlib import Path
from pprint import pformat

import numpy as np
from sklearn import metrics
from svc import EmotionRecSVC
from utils import (SummaryWriter, load_data, plot_confusion_matrix,
                   plot_sample_predictions)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
DATASET_DIR = str(PROJECT_DIR.joinpath("cw_dataset"))
MODELS_DIR = str(PROJECT_DIR.joinpath("models"))
RUN_NAME = f"SIFT-SVC_{time.strftime('%Y-%m-%d %H-%M')}"


def main(args):
    writer = SummaryWriter(log_dir=str(PROJECT_DIR) + f"/logs/{RUN_NAME}")
    logger.info(f"Input args: {pformat(args)}")
    X_train, y_train = load_data(
        DATASET_DIR,
        "train",
        hog_dict=dict(),
        batch_size=None,
        method="sift"
    )
    X_val, y_val = load_data(
        DATASET_DIR,
        "val",
        hog_dict=dict(),
        batch_size=None,
        method="sift"
    )
    model = EmotionRecSVC(
        kernel=args.kernel,
        C=args.c_constant,
        cluster_factor=args.cluster_factor,
        batch_divisor=args.batch_divisor,
    )
    model.fit_transform(X_train, y_train)

    model_name = f"{RUN_NAME}.joblib"
    model_file_name = MODELS_DIR + "/" + model_name
    joblib.dump(model, model_file_name)

    predicted = model.predict(X_val)

    labels = np.unique(np.concatenate([y_val, predicted], axis=0))
    plot_confusion_matrix(y_val, predicted, labels, model="SIFT-SVC", no_preds=True, writer=writer)
    # load it in again to get the images instead of SIFT feature descriptors
    X_val, _ = load_data(
        DATASET_DIR,
        "val",
        "normal",
        hog_dict=dict(),
        batch_size=None,
        shuffle=False,
        drop_last=False,
        weighted_sampling=True,
    )

    metrics_dict = dict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_sample_predictions(X_val, predicted, y_val, 4, 5, "SIFT-SVC", writer=writer)
        logger.info(print(metrics.classification_report(y_val, predicted)))
        metrics_dict["accuracy"] = metrics.accuracy_score(y_val, predicted) * 100
        metrics_dict["recall"] = metrics.recall_score(y_val, predicted, average="weighted")
        metrics_dict["precision"] = metrics.precision_score(y_val, predicted, average="weighted")
        metrics_dict["f1_score"] = metrics.f1_score(y_val, predicted, average="weighted")
    logger.info(str(metrics_dict))

    writer.add_hparams(
        hparam_dict={
            "Pixels Per Cell": 0,
            "Orientation": 0,
            "Kernel": args.kernel,
            "C": args.c_constant,
            "Cluster Factor": args.cluster_factor,
            "Batch Divisor": args.batch_divisor,
            "Dropout Rate": 0,
            "Learn Rate Scheduler": None,
            "Batch Size": 0,
            "Max Learning Rate": 0,
            "Epochs": 0,
            "Optimizer": None,
            "Weight Decay": 0,
        },
        metric_dict=metrics_dict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--kernel",
        default="rbf",
        type=str,
        help="Type of kernel for SVM",
        choices=["rbf", "poly"],
    )
    parser.add_argument(
        "-c",
        "--c-constant",
        default=1,
        type=float,
        help="C parameter for SVM tuning"
    )
    parser.add_argument(
        "-cf",
        "--cluster-factor",
        default=30,
        type=int,
        help="Determines number of clusters. K = unique classes * cluster factor",
    )
    parser.add_argument(
        "-bd",
        "--batch-divisor",
        default=10,
        type=int,
        help="Determines batch size for MiniKmeans algorithm. Num samples // batch-divisor",
    )
    args = parser.parse_args()
    main(args)