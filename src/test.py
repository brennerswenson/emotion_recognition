from cnn import EmotionRecCNN
from mlp import EmotionRecMLP
from matplotlib import pyplot as plt
import logging
import os
from pathlib import Path
import joblib
from sklearn import metrics
from PIL import Image
import numpy as np
import random
import cv2
import torch
from config import LABELS
from tqdm import tqdm

from collections import OrderedDict

from utils import load_data, plot_sample_predictions, get_pred_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
DATASET_DIR = str(PROJECT_DIR.joinpath("cw_dataset"))
MODELS_DIR = str(PROJECT_DIR.joinpath("models"))
VIDEOS_DIR = str(PROJECT_DIR.joinpath("video"))

SVM_MODEL = "SIFT-SVC_2021-04-12 22-08.joblib"
CNN_MODEL = "CNN_2021-04-14 14-38.pth"
MLP_MODEL = "MLP_2021-04-12 19-52.pth"

MLP_INPUT = 10000  # length of hog feature descriptors
MLP_HOG_DICT = {"orientation": 16, "pix_per_cell": (4, 4)}
MLP_BATCH_SIZE = 256
CNN_BATCH_SIZE = 32
SVM_BATCH_SIZE = 256


class EmotionRecognition:
    def __init__(self, test_path=None, model_type=None):
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
            self.is_img = True
            self.model = EmotionRecCNN(output_size=7, dropout_rate=0.5)
            self.model.load_state_dict(torch.load(f"{MODELS_DIR}/{CNN_MODEL}", map_location="cpu"))
            self.model.eval()

        elif self.model_type == "MLP":
            self.is_img = False
            self.model = EmotionRecMLP(
                MLP_INPUT, MLP_INPUT // 2, MLP_INPUT // 4, MLP_INPUT // 8, 7
            )
            self.model.load_state_dict(torch.load(f"{MODELS_DIR}/{MLP_MODEL}", map_location="cpu"))
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
                DATASET_DIR,
                "test",
                "cnn",
                hog_dict=dict(),
                batch_size=CNN_BATCH_SIZE,
                shuffle=False,
            )

        elif self.model_type == "MLP":
            self.data_loader = load_data(
                DATASET_DIR,
                "test",
                "hog",
                hog_dict=MLP_HOG_DICT,
                batch_size=MLP_BATCH_SIZE,
                shuffle=False,
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
            metrics_dict["precision"] = metrics.precision_score(
                self.y, predictions, average="weighted"
            )
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
                accuracy=metrics_dict["accuracy"],
            )
        return predictions, metrics_dict


class EmotionRecognitionVideo(EmotionRecognition):
    def __init__(self, model_type):
        super(EmotionRecognitionVideo, self).__init__(model_type=model_type)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )

        self.video_path = None
        self.height = None
        self.width = None
        self.fps = None

    def predict_video(self, video_path):
        self.video_path = video_path
        frames = self.get_frames()
        face_dict = self.get_faces(frames)
        face_dict = self.get_predictions(face_dict)
        frames = self.annotate_frames(frames, face_dict)
        self.save_video(frames)

    def save_video(self, frames):
        logger.info("Saving video")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(
            self.video_path.split(".")[0] + "_OUTPUT.mp4",
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_video.write(frame)
        output_video.release()

    def get_frames(self):
        logger.info("Reading in video")
        video = cv2.VideoCapture(self.video_path)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        frames = list()
        while video.isOpened():
            ret, frame = video.read()

            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(img)

                # set an attribute for width and height
                if self.height is None:
                    self.height = frame.shape[0]
                    self.width = frame.shape[1]

            else:
                break
        video.release()
        cv2.destroyAllWindows()
        return frames

    def get_faces(self, frames):
        self.min_size = self.height // 14
        profile_face_tolerance = 0.75
        profe_face_high = 1 + profile_face_tolerance
        profe_face_low = 1 - profile_face_tolerance

        logger.info(f"Finding faces with min size {self.min_size} x {self.min_size}")
        face_dict = OrderedDict()
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            face_boxes = self.face_cascade.detectMultiScale(
                frame, 1.03, 35, minSize=(self.min_size, self.min_size)
            )
            profile_boxes = self.profile_cascade.detectMultiScale(
                frame, 1.03, 35, minSize=(self.min_size, self.min_size)
            )
            logger.info(f"{len(face_boxes)} faces")
            logger.info(f"{len(profile_boxes)} profiles")

            if isinstance(face_boxes, np.ndarray) and isinstance(profile_boxes, np.ndarray):
                similarity_scores = [(x, y, x / y) for y in profile_boxes for x in face_boxes]
                profile_boxes = np.array([
                    x[1]
                    for x in similarity_scores
                    if not (
                        (profe_face_low <= x[2][0] <= profe_face_high)
                        or (profe_face_low <= x[2][1] <= profe_face_high)
                    )
                ])
                if profile_boxes.any():
                    logger.info(f"Adding {len(profile_boxes)} profile not present in faces")
                    face_boxes = np.concatenate((face_boxes, profile_boxes), axis=0)
            if isinstance(profile_boxes, np.ndarray) and isinstance(face_boxes, tuple):
                face_boxes = profile_boxes

            logger.info(f"{len(face_boxes)} objects")

            if isinstance(face_boxes, np.ndarray):
                face_images = [self.extract_face(frame, face_arr) for face_arr in face_boxes]
                plt.imshow(random.choice(face_images))
                plt.show()
                face_dict[idx] = list(zip(face_boxes, face_images))
            else:
                face_dict[idx] = list()
        return face_dict

    def get_predictions(self, face_dict):
        logger.info("Beginning predictions")
        pred_face_dict = dict()
        for frame, face_arr in tqdm(face_dict.items(), total=len(face_dict)):
            if frame:
                pred_face_arr = list()
                for i, (face_box, face_img) in enumerate(face_arr):
                    face_img = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_AREA)
                    face_tensor = torch.FloatTensor(face_img).unsqueeze(0)
                    face_tensor = face_tensor.permute(0, 3, 1, 2)

                    outputs = self.model(face_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    pred_label = LABELS[predicted.item()]
                    pred_face_arr.append((face_box, face_img, pred_label))
                pred_face_dict[frame] = pred_face_arr
            else:
                pred_face_dict[frame] = list()
        return pred_face_dict

    @staticmethod
    def annotate_frames(frames, face_dict):
        logger.info("Annotating video frames")
        annotated_frames = list()
        for frame, (frame_idx, face_arr) in zip(frames, face_dict.items()):
            if face_arr:
                for i, face in enumerate(face_arr):
                    if face:
                        x, y, w, h = face[0]
                        face_label = face[2]
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (32, 178, 170), 3)
                        frame = cv2.putText(
                            frame,
                            face_label,
                            (x - 20, y - 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            5,
                            (32, 178, 170),
                            6,
                        )
            annotated_frames.append(frame)
        return frames

    @staticmethod
    def extract_face(frame, index_arr):
        y = index_arr[1]
        x = index_arr[0]
        rect_width = index_arr[2]
        rect_height = index_arr[3]
        extracted = frame[y : y + rect_height, x : x + rect_width]
        return extracted


if __name__ == "__main__":
    # em = EmotionRecognition(DATASET_DIR + "/test", model_type="CNN")
    # preds, metrics = em.predict_all(visualise=True)
    # logger.info(f"{str(metrics)}")

    erv = EmotionRecognitionVideo(model_type="CNN")
    frames = erv.predict_video(VIDEOS_DIR + "\pexels-rodnae-productions-5617899.mp4")
    # frames = erv.predict_video(VIDEOS_DIR + '\pexels-diva-plavalaguna-6194825.mp4')
    # frames = erv.predict_video(VIDEOS_DIR + '\pexels-diva-plavalaguna-6194803.mp4')
    # frames = erv.predict_video(VIDEOS_DIR + "\WIN_20210413_20_36_01_Pro.mp4")
    # frames = erv.predict_video(VIDEOS_DIR + '\pexels-diva-plavalaguna-6194825.mp4')
    # frames = erv.predict_video(VIDEOS_DIR + '\pexels-rodnae-productions-5617899.mp4')
