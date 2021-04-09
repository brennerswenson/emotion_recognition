import logging
import time

import numpy as np
import seaborn as sns
from config import LABELS
from matplotlib import pyplot as plt
from sklearn import metrics
from SVC import EmotionRecSVC
from utils import load_data, plot_sample_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    KERNEL = "rbf"
    C = 5
    CLUSTER_FACTOR = 40
    BATCH_DIV = 8
    X_train, y_train = load_data(
        "../CW_Dataset",
        "train",
        hog_dict=dict(),
        batch_size=None,
        method="normal"
    )
    X_val, y_val = load_data(
        "../CW_Dataset",
        "val",
        hog_dict=dict(),
        batch_size=None,
        method="normal"
    )
    start = time.time()
    model = EmotionRecSVC(
        KERNEL,
        C=C,
        cluster_factor=CLUSTER_FACTOR,
        batch_divisor=BATCH_DIV,
    )
    model.fit_transform(X_train, y_train)
    end = time.time()

    elapsed = end - start
    predicted = model.predict(X_val)

    labels = np.unique(np.concatenate([y_val, predicted], axis=0))
    cm = metrics.confusion_matrix(
        y_val,
        predicted,
        labels=labels,
    )
    f = sns.heatmap(
        cm,
        annot=True,
        xticklabels=list(LABELS.values()),
        yticklabels=list(LABELS.values()),
        fmt="g",
    )
    plt.show()

    accuracy = [y_p == y_t for y_p, y_t in zip(predicted, y_val)]
    accuracy = sum(accuracy) / len(y_val) * 100

    results = dict()
    results["kernel"] = KERNEL
    results["batch_divisor"] = BATCH_DIV
    results["C"] = C
    results["cluster_factor"] = BATCH_DIV
    results["elapsed"] = elapsed
    results["accuracy"] = accuracy
    results["recall"] = metrics.recall_score(y_val, predicted, average="weighted")
    results["precision"] = metrics.precision_score(y_val, predicted, average="weighted")
    results["f1_score"] = metrics.f1_score(y_val, predicted, average="weighted")

    logger.info(str(results))
    logger.info(print(metrics.classification_report(y_val, predicted)))
    plot_sample_predictions(X_val, predicted, y_val, 4, 5, "SIFT-SVM")
    plt.show()
