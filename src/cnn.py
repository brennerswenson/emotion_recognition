
import torch.nn as nn
from utils import BaseModel


class EmotionRecCNN(BaseModel):
    def __init__(self, output_size, dropout_rate):
        super(EmotionRecCNN, self).__init__()

        self.dropout_rate = dropout_rate
        self.output_size = output_size

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # output: 64 x 50 x 50
            nn.Dropout(self.dropout_rate),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # output: 128 x 25 x 25
            nn.Dropout(self.dropout_rate),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),  # output :256x 25 x 25
            nn.BatchNorm2d(256),
            nn.MaxPool2d(5, 5),  # output: 256 x 5 x 5
            nn.Dropout(self.dropout_rate),

            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1024),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(512, self.output_size),
        )

    def forward(self, x):
        return self.network(x)
