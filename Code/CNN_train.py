import logging

from CNN import EmotionRecCNN
from utils import (
    DLDevice,
    eval_model,
    get_avail_device,
    load_data,
    send_to_device,
    train_model,
    plot_sample_predictions,
    get_pred_metrics,
    plot_confusion_matrix,
)

import torch
import time
from sklearn import metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    batch_size = 64
    epochs = 20
    lr_max = 1e-3
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    train_dl = load_data(
        "../CW_Dataset",
        "train",
        "cnn",
        hog_dict=dict(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        weighted_sampling=True,
    )

    val_dl = load_data(
        "../CW_Dataset",
        "val",
        "cnn",
        hog_dict=dict(),
        batch_size=batch_size
    )

    num_samples = len(train_dl.dataset.imgs)
    device = get_avail_device()
    train_dl = DLDevice(train_dl, device)
    val_dl = DLDevice(val_dl, device)
    model = EmotionRecCNN(output_size=num_samples)
    model = send_to_device(model, device)
    hist = train_model(
        epochs,
        lr_max,
        model,
        train_dl,
        val_dl,
        weight_decay=weight_decay,
        optimizer=opt_func,
    )
    logger.info((eval_model(model, val_dl)))

    all_preds, y_val, X_val, metrics_dict = get_pred_metrics(model, val_dl, device)
    logger.info(metrics_dict)
    logger.info(print(metrics.classification_report(y_val.cpu(), all_preds.cpu())))
    unique_labels = [int(x) - 1 for x in val_dl.dataset.classes]
    plot_confusion_matrix(y_val, all_preds, unique_labels, no_preds=False)
    plot_sample_predictions(X_val.cpu(), all_preds.cpu(), y_val, 4, 5, "CNN")

    model_path = "../Models"
    model_name = f"CNN_{time.strftime('%Y-%m-%d %H-%S')}.pth"

    model_file_name = model_path + "/" + model_name
    torch.save(model.state_dict(), model_file_name)
