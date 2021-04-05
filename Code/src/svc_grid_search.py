import logging
import time

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

from SVC import EmotionRecSVC
from config import LABELS
from utils import load_data, plot_sample_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    X, y = load_data("../../CW_Dataset", "train")

    kernels = ["rbf", "poly"]
    C_options = [0.1, 0.5, 0.8, 1, 1.2, 1.5, 2, 5, 10]
    cluster_factors = [10, 20, 30]
    batch_div_opts = [2, 4, 6, 8]

    total_combos = len(kernels) * len(C_options) * len(cluster_factors) * len(batch_div_opts)
    search_results = list()

    counter = 0

    for kernel in kernels:
        for c in C_options:
            for cluster_fact in cluster_factors:
                for b_div in batch_div_opts:
                    logger.info(f"{counter + 1} out of {total_combos}")

                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

                    results = dict()

                    start = time.time()
                    model = EmotionRecSVC(
                        kernel,
                        C=c,
                        cluster_factor=cluster_fact,
                        batch_divisor=b_div,
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

                    results["kernel"] = kernel
                    results["batch_divisor"] = b_div
                    results["C"] = c
                    results["cluster_factor"] = cluster_fact
                    results["elapsed"] = elapsed
                    results["accuracy"] = accuracy
                    results["recall"] = metrics.recall_score(y_val, predicted, average="weighted")
                    results["precision"] = metrics.precision_score(y_val, predicted, average="weighted")
                    results["f1_score"] = metrics.f1_score(y_val, predicted, average="weighted")

                    counter += 1

                    search_results.append(results)
                    logger.info(str(results))

                    logger.info(print(metrics.classification_report(y_val, predicted)))
                    if counter % 5 == 0:
                        df = pd.DataFrame.from_records(search_results)
                        df.to_csv(
                            f"../../Outputs/SVC_grid_{time.strftime('%Y-%m-%d %H-%S')}.csv",
                            index=False,
                        )

                    plot_sample_predictions(X_val, predicted, y_val, 10)
                    plt.show()
