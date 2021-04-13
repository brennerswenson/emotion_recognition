from cnn import EmotionRecCNN
from mlp import EmotionRecMLP
import torch
import logging
import os
from pathlib import Path
import joblib
from sklearn import metrics
from PIL import Image
import numpy as np

from utils import load_data, plot_sample_predictions, get_pred_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
DATASET_DIR = str(PROJECT_DIR.joinpath("cw_dataset"))
MODELS_DIR = str(PROJECT_DIR.joinpath("models"))

SVM_MODEL = "SIFT-SVC_2021-04-12 22-08.joblib"
CNN_MODEL = "CNN_2021-04-12 19-43.pth"
MLP_MODEL = "MLP_2021-04-12 19-52.pth"

MLP_INPUT = 10000  # length of hog feature descriptors
MLP_HOG_DICT = {"orientation": 16, "pix_per_cell": (4, 4)}
MLP_BATCH_SIZE = 256
CNN_BATCH_SIZE = 32
SVM_BATCH_SIZE = 256


class EmotionRecognition:
    def __init__(self, test_path, model_type):
        self.test_path = test_path
        self.model_type = model_type

        self.is_img = None
        self.model = None
        self.X = None
        self.y = None
        self.data_loader = None

        self.get_model()

    def get_model(self):

        if self.model_type == "SVM":
            self.is_img = False
            self.model = joblib.load(f"{MODELS_DIR}/{SVM_MODEL}")

        elif self.model_type == "CNN":
            self.is_img= True
            self.model = EmotionRecCNN(output_size=7, dropout_rate=0.5)
            self.model.load_state_dict(torch.load(f"{MODELS_DIR}/{CNN_MODEL}", map_location='cpu'))
            self.model.eval()

        elif self.model_type == 'MLP':
            self.is_img = False
            self.model = EmotionRecMLP(
                MLP_INPUT, MLP_INPUT // 2, MLP_INPUT // 4, MLP_INPUT // 8, 7
            )
            self.model.load_state_dict(torch.load(f"{MODELS_DIR}/{MLP_MODEL}", map_location='cpu'))
            self.model.eval()
        else:
            logger.info("Invalid Model Type")

    def get_data(self):
        if self.model_type == "SVM":
            X_test, y_test = load_data(
                DATASET_DIR, "test", hog_dict=dict(), batch_size=None, method="sift", shuffle=False
            )
            self.X = X_test
            self.y = y_test
        elif self.model_type == "CNN":
            self.data_loader = load_data(
                DATASET_DIR, "test", "cnn", hog_dict=dict(), batch_size=CNN_BATCH_SIZE, shuffle=False,
            )

        elif self.model_type == "MLP":
            self.data_loader = load_data(
                DATASET_DIR, "test", "hog", hog_dict=MLP_HOG_DICT, batch_size=MLP_BATCH_SIZE, shuffle=False
            )

        else:
            logger.info("Invalid Model Type")

    def predict_all(self, visualise=False):
        self.get_data()
        if self.model_type == "SVM":
            predictions = self.model.predict(self.X)
            self.X, self.y = load_data(
                DATASET_DIR,
                "test",
                "normal",
                hog_dict=dict(),
                batch_size=SVM_BATCH_SIZE,
                shuffle=False,
                drop_last=False,
                weighted_sampling=False,
            )
            metrics_dict = dict()
            metrics_dict["accuracy"] = metrics.accuracy_score(self.y, predictions) * 100
            metrics_dict["recall"] = metrics.recall_score(self.y, predictions, average="weighted")
            metrics_dict["precision"] = metrics.precision_score(self.y, predictions, average="weighted")
            metrics_dict["f1_score"] = metrics.f1_score(self.y, predictions, average="weighted")

        elif self.model_type == "CNN":
            predictions, self.y, self.X, metrics_dict = get_pred_metrics(
                self.model, self.data_loader, "cpu"
            )
        elif self.model_type == "MLP":
            predictions, self.y, _, metrics_dict = get_pred_metrics(
                self.model, self.data_loader, "cpu"
            )
            # load in images instead of HOG feature descriptors for viz
            self.X = [Image.open(x[0]) for x in self.data_loader.dataset.imgs]
            self.X = [np.asarray(x) for x in self.X]

        if visualise:
            plot_sample_predictions(
                self.X,
                predictions,
                self.y,
                2,
                2,
                self.model_type,
                tensor=self.is_img,
                figsize=(5, 5),
                accuracy=metrics_dict['accuracy']
            )
        return predictions, metrics_dict


if __name__ == '__main__':
    em = EmotionRecognition(DATASET_DIR + "/test", model_type="SVM")
    preds, metrics = em.predict_all(visualise=True)
    logger.info(f"{str(metrics)}")
