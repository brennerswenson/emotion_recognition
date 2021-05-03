from cnn import EmotionRecCNN
from mlp import EmotionRecMLP
from matplotlib import pyplot as plt
import logging
import os
from pathlib import Path
import joblib
from sklearn import metrics
import numpy as np
import random
import cv2
import torch
from config import LABELS
from tqdm import tqdm

from collections import OrderedDict

from utils import load_data, plot_sample_predictions, get_pred_metrics, plot_confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
DATASET_DIR = str(PROJECT_DIR.joinpath("cw_dataset"))
MODELS_DIR = str(PROJECT_DIR.joinpath("models"))
VIDEOS_DIR = str(PROJECT_DIR.joinpath("video"))

SVM_MODEL = "SIFT-SVC_2021-04-14 00-20.joblib"
CNN_MODEL = "CNN_2021-04-29 10-11.pth"
MLP_MODEL = "MLP_2021-05-01 14-02.pth"

MLP_INPUT = 14112  # length of hog feature descriptors
MLP_HOG_DICT = {"orientation": 8, "pix_per_cell": (6, 6)}


class EmotionRecognition:
    """Class to classify emotions found in images using a variety of methods."""
    def __init__(self, test_path=None, model_type=None, batch_size=1):
        """

        Args:
            test_path (str): Full path to the directory containing the testing data.
            model_type (str): Acronym denoting the type of model to load/run. SVM, CNN, or MLP.
            batch_size (int): For MLP and CNN models, the batch size used when predicting.
                    For testing individual images use a batch size of 1.
        """
        self.test_path = test_path
        self.model_type = model_type
        self.batch_size = batch_size

        self.is_img = None
        self.model = None
        self.X = None
        self.y = None
        self.data_loader = None
        self.file_paths = None

        self._get_model()

    def _get_model(self):
        """Depending on the model passed to the constructor, load the trained model and save to class attribute."""
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
                MLP_INPUT, MLP_INPUT // 3, MLP_INPUT // 6, MLP_INPUT // 9, 7
            )
            self.model.load_state_dict(torch.load(f"{MODELS_DIR}/{MLP_MODEL}", map_location="cpu"))
            self.model.eval()
        else:
            logger.info("Invalid Model Type")

    def _get_data(self, num_images):
        """
        Load the testing data. This can either be a DataLoader or arrays depending on the model type.
        Args:
            num_images (int): Number of images to randomly create predictions for. If not provided, all images
                    in the testing dataset will be predicted.

        Returns: None

        """
        total_img = num_images if num_images is not None else "all"
        logger.info(f"Loading test data for {total_img} images")
        shuffle = True if num_images is not None else False
        if self.model_type == "SVM":
            X_test, y_test, self.file_paths = load_data(
                DATASET_DIR,
                "test",
                hog_dict=dict(),
                batch_size=None,
                method="sift",
                shuffle=shuffle,
                num_images=num_images,
            )
            self.X = X_test
            self.y = y_test
        elif self.model_type == "CNN":
            self.data_loader = load_data(
                DATASET_DIR,
                "test",
                "cnn",
                hog_dict=dict(),
                batch_size=self.batch_size,
                shuffle=shuffle,
            )

        elif self.model_type == "MLP":
            self.data_loader = load_data(
                DATASET_DIR,
                "test",
                "hog",
                hog_dict=MLP_HOG_DICT,
                batch_size=self.batch_size,
                shuffle=shuffle,
            )

        else:
            logger.info("Invalid Model Type")

    def predict(self, visualise_samples=False, num_test_images=None, vis_confusion_matrix=False):
        """
        Predict on a subset of images or entire test dataset. Metrics are calculated like F1
        score, recall, precision, etc. for predictions. If num_test_images not provided,
        predictions are made for entire test dataset.
        Args:
            visualise_samples (bool): Indicates if sample predictions should be plotted or not.
            num_test_images (int): Number of test images to predict. If not supplied,
                    all test images will be predicted.

        Returns: (tuple) Predictions, metric_dict

        """
        logger.info("Making predictions on test data")
        self._get_data(num_images=num_test_images)
        if self.model_type == "SVM":
            predictions = self.model.predict(self.X)

            self.X = [cv2.imread(x) for x in self.file_paths]
            self.X = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in self.X]
            self.X = [np.asarray(x) for x in self.X]

            metrics_dict = dict()
            metrics_dict["accuracy"] = metrics.accuracy_score(self.y, predictions) * 100
            metrics_dict["recall"] = metrics.recall_score(self.y, predictions, average="weighted")
            metrics_dict["precision"] = metrics.precision_score(
                self.y, predictions, average="weighted"
            )
            metrics_dict["f1_score"] = metrics.f1_score(self.y, predictions, average="weighted")

        elif self.model_type == "CNN":
            predictions, self.y, self.X, metrics_dict, file_paths = get_pred_metrics(
                self.model, self.data_loader, "cpu", num_images=num_test_images
            )
        elif self.model_type == "MLP":
            predictions, self.y, _, metrics_dict, file_paths = get_pred_metrics(
                self.model, self.data_loader, "cpu", num_images=num_test_images
            )
            # load in images instead of HOG feature descriptors for viz
            self.X = [cv2.imread(x) for x in file_paths]
            self.X = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in self.X]
            self.X = [np.asarray(x) for x in self.X]

        if visualise_samples:
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

        if vis_confusion_matrix:
            unique_labels = [int(x) - 1 for x in self.data_loader.dataset.classes]
            plot_confusion_matrix(self.y, predictions, unique_labels, self.model_type)

        return predictions, metrics_dict


class EmotionRecognitionVideo(EmotionRecognition):
    """Class to classify emotions found in video frames using a variety of approaches."""
    def __init__(self, model_type):
        """

        Args:
             model_type (str): Acronym denoting the type of model to load/run. SVM, CNN, or MLP.
        """
        super(EmotionRecognitionVideo, self).__init__(model_type=model_type)
        # set haarcascade models as class attributes
        # one model is for frontal face detection
        # the other is for profile face detection
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
        """
        Find and classify emotions in each video frame, then save video.
        Args:
            video_path (str): File path to the video to be annotated with emotions.

        Returns: None

        """
        self.video_path = video_path
        frames = self._get_frames()
        face_dict = self._get_faces(frames)
        face_dict = self._get_predictions(face_dict)
        frames = self._annotate_frames(frames, face_dict)
        self._save_video(frames)

    def _save_video(self, frames):
        """
        Iterate through annotated frames and write the frame to a video file. Output is
        saved in the same directory as the original video file.
        Args:
            frames (list): List of frames with annotated emotions and bounding boxes.

        Returns: None

        """
        logger.info("Saving video")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        file_path = self.video_path.split(".")[0] + "_OUTPUT.mp4"
        output_video = cv2.VideoWriter(
            file_path,
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        logger.info(f"Saved output to {file_path}")
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_video.write(frame)
        output_video.release()

    def _get_frames(self):
        """Load in video from video path and read in frames to a list."""
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

    def _get_faces(self, frames):
        """
        Iterate through all video frames and find faces in each frame. Find all of the portrait
        oriented faces first, then if there is no face found, fill in with side-profile faces that were
        found for the same frame.
        Args:
            frames (list of np.arrays): List containing all video frames.

        Returns (dict): Ordered dictionary containing all of the faces found in each frame.

        """
        self.min_size = self.height // 10
        profile_face_tolerance = .25
        profe_face_high = 1 + profile_face_tolerance
        profe_face_low = 1 - profile_face_tolerance

        logger.info(f"Finding faces with min size {self.min_size} x {self.min_size}")
        face_dict = OrderedDict()
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            # detect faces using both models
            face_boxes = self.face_cascade.detectMultiScale(
                frame, 1.04, 30, minSize=(self.min_size, self.min_size)
            )
            profile_boxes = self.profile_cascade.detectMultiScale(
                frame, 1.04, 30, minSize=(self.min_size, self.min_size)
            )
            logger.info(f"{len(face_boxes)} faces")
            logger.info(f"{len(profile_boxes)} profiles")

            # if we found portrait faces and profile faces
            if isinstance(face_boxes, np.ndarray) and isinstance(profile_boxes, np.ndarray):
                # if they found the same faces, then we want to keep the portrait face and discard profile
                # divide the bounding box coordinates of profile faces by portrait faces
                # if the scores are between, for example, 1.7 and 0.3, then they are likely the same face.
                similarity_scores = [(x, y, x / y) for y in profile_boxes for x in face_boxes]
                # discard the profile faces that match the portrait faces
                profile_boxes = np.array(
                    [
                        x[1]
                        for x in similarity_scores
                        if not (
                            (profe_face_low <= x[2][0] <= profe_face_high)
                            or (profe_face_low <= x[2][1] <= profe_face_high)
                        )
                    ]
                )
                profile_boxes = np.array([np.array(list(x)) for x in set(tuple(x) for x in profile_boxes)])
                if profile_boxes.any():
                    logger.info(f"Adding {len(profile_boxes)} profile not present in faces")
                    face_boxes = np.concatenate((face_boxes, profile_boxes), axis=0)
            # if profiles are found and no portrait faces are found, set face_boxes = profile_boxes
            if isinstance(profile_boxes, np.ndarray) and isinstance(face_boxes, tuple):
                face_boxes = profile_boxes

            logger.info(f"{len(face_boxes)} objects")

            # if we have found any faces
            if isinstance(face_boxes, np.ndarray):
                # extract the face images from the video frames and add to return array at index idx
                face_images = [self._extract_face(frame, face_arr) for face_arr in face_boxes]
                # fig, axes = plt.subplots(1, 2)
                # axes = axes.ravel()
                # axes[0].imshow(random.choice(face_images))
                # axes[1].imshow(frame)
                # plt.show()
                face_dict[idx] = list(zip(face_boxes, face_images))
            else:
                face_dict[idx] = list()
        return face_dict

    def _get_predictions(self, face_dict):
        """
        Iterate through the dictionary containing all of the faces, and classify the emotions
        of each face. The faces need to be resized to 100x100. Returns a dictionary containing
        the coordinates, the face image, and the label.
        Args:
            face_dict (dict): Ordered dictionary containing all of the faces found in each frame.

        Returns (dict): Ordered dictionary containing lists of the face coordinates, images, and predictions
        """
        logger.info("Beginning predictions")
        pred_face_dict = OrderedDict()
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
    def _annotate_frames(frames, face_dict):
        """
        Iterate through all video frames, place bounding box on faces, and write text
        of predicted emotion above the bounding box of each face.
        Args:
            frames (list of np.arrays): List containing all video frames.
            face_dict (dict): Ordered dictionary containing lists of the face coordinates, images, and predictions

        Returns (list): List of frames with annotated emotions and bounding boxes.

        """
        logger.info("Annotating video frames")
        annotated_frames = list()
        for frame, (frame_idx, face_arr) in zip(frames, face_dict.items()):
            if face_arr:
                for i, face in enumerate(face_arr):
                    if face:
                        x, y, w, h = face[0]
                        face_label = face[2]
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (32, 178, 170), 5)
                        frame = cv2.putText(
                            frame,
                            face_label,
                            (x - 20, y - 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            6,
                            (32, 178, 170),
                            9,
                        )
            annotated_frames.append(frame)
        return frames

    @staticmethod
    def _extract_face(frame, index_arr):
        """
        Using coordinates from haarcascades, extract the face from the given frame and
        return the extracted image.
        Args:
            frame (np.array): Video frame with faces to extract
            index_arr (np.array): Array containing the coordinates and height/width of face to extract.

        Returns (np.array) Slice of video frame with detected face.

        """
        y = index_arr[1]
        x = index_arr[0]
        rect_width = index_arr[2]
        rect_height = index_arr[3]
        extracted = frame[y: y + rect_height, x: x + rect_width]
        return extracted
