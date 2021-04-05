import logging
import time

import torch
from torch.utils import data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch.optim as optim

from torch import nn

from utils import load_data, HOG
from MLP import EmotionRecMLP

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train, y_train = load_data("../../CW_Dataset", "train", mode="sklearn")

    hog = HOG(
        orientations=8,
        pix_per_cell=(3, 3),
        cells_per_block=(1, 1),
        multichannel=True
    )
    hog_arr = hog.fit_transform(X_train)

    tensor_X_train = torch.tensor(hog_arr).float()
    tensor_y_train = torch.tensor(y_train)

    train_dataset = data.TensorDataset(tensor_X_train, tensor_y_train)
    my_dataloader = data.DataLoader(train_dataset, batch_size=2)

    t0 = time.time()

    net = EmotionRecMLP(hog_arr[0].shape[0], 2500, 1200, 500, 7)
    net = net.float()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(4):  # loop over the training set two times

        running_loss = 0.0
        for i, data_batch in enumerate(my_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data_batch[0].to(device), data_batch[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics (loss.item() returns the mean loss in the mini-batch)
            running_loss += loss.item()
            if i % 32 == 0:
                logger.info(
                    "[%d, %5d] loss: %.5f" % (epoch + 1, i + 1, running_loss / 32)
                )
                running_loss = 0.0

    logger.info(
        f"Finished Training: total time in seconds = {time.time() - t0}",
    )
    model_path = "..\\..\\Models"
    model_name = f"hog_mlp_{time.strftime('%Y-%m-%d %H-%S')}.pth"

    model_file_name = model_path + "/" + model_name
    torch.save(net.state_dict(), model_file_name)

    net = EmotionRecMLP(hog_arr[0].shape[0], 2500, 1200, 500, 7)
    net.load_state_dict(torch.load(model_file_name))

    X_val, y_val = load_data("../../CW_Dataset", "val", mode="sklearn")

    hog_arr_val = hog.fit_transform(X_val)
    tensor_X_val = torch.tensor(hog_arr_val).float()
    tensor_y_val = torch.tensor(y_val)
    val_dataset = data.TensorDataset(tensor_X_val, tensor_y_val)
    val_dataloader = data.DataLoader(val_dataset, batch_size=2, shuffle=True)

    # Estimate average accuracy
    correct = 0
    total = 0
    with torch.no_grad():  # Avoid backprop at test
        for data_batch in val_dataloader:
            images, labels = data_batch
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info(
        f"Accuracy of the network on the validation images: {100 * correct / total}%"
    )
