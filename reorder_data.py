import numpy as np
import pandas as pd
import logging
import shutil
from pathlib import Path
import os

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = "cw_dataset"
subsets = ["train", "test"]


if __name__ == "__main__":
    for subset in subsets:
        file_list = np.genfromtxt(path + f"/labels/list_label_{subset}.txt", dtype=str)
        file_df = pd.DataFrame(file_list, columns=["file_name", "label"])

        if subset == "train":
            train_df, val_df, train_labels, val_labels = train_test_split(
                file_df["file_name"],
                file_df["label"],
                test_size=0.2,
                shuffle=True,
                stratify=file_df["label"],
            )

            train_df, val_df = pd.DataFrame(train_df), pd.DataFrame(val_df)

            train_df["label"] = train_labels
            val_df["label"] = val_labels

            train_val_list = [(train_df, "train"), (val_df, "val")]

            for sub_df, subset in train_val_list:
                for label in sub_df["label"].unique():
                    Path(f"{path}/{subset}/{label}").mkdir(parents=True, exist_ok=True)
                    label_imgs = sub_df.loc[
                        sub_df["label"] == label, "file_name"
                    ].to_list()
                    label_img_paths = [
                        os.path.join(
                            os.getcwd(),
                            path,
                            "train",
                            x.split(".jpg")[0] + "_aligned.jpg",
                        )
                        for x in label_imgs
                    ]

                    for img_path in label_img_paths:
                        try:
                            shutil.move(
                                img_path, os.path.join(os.getcwd(), path, subset, label)
                            )
                        except Exception as e:
                            logger.info(e)

        elif subset == "test":
            for label in file_df["label"].unique():
                Path(f"{path}/{subset}/{label}").mkdir(parents=True, exist_ok=True)
                label_imgs = file_df.loc[
                    file_df["label"] == label, "file_name"
                ].to_list()
                label_img_paths = [
                    os.path.join(
                        os.getcwd(), path, subset, x.split(".jpg")[0] + "_aligned.jpg"
                    )
                    for x in label_imgs
                ]

                for img_path in label_img_paths:
                    try:
                        shutil.move(
                            img_path, os.path.join(os.getcwd(), path, subset, label)
                        )
                    except Exception as e:
                        logger.info(e)
