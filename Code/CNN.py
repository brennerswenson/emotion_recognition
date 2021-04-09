import torch.nn as nn
from utils import BaseModel

import torch.nn.functional as F


class EmotionRecCNN(BaseModel):
    def __init__(self, output_size):
        super(EmotionRecCNN, self).__init__()

        self.output_size = output_size

        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(12, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, self.output_size)

        # self.conv_1 = self.convolution_block(3, 128)  # 3 x 64 x 64
        # self.conv_2 = self.convolution_block(128, 256, pool=True)  # 128 x 32 x 32
        # self.res_1 = nn.Sequential(
        #     self.convolution_block(256, 256), self.convolution_block(256, 256)
        # )  # 256 x 32 x 32
        #
        # self.conv_3 = self.convolution_block(256, 512, pool=True)  # 512 x 16 x 16
        # self.conv_4 = self.convolution_block(512, 1024, pool=True)  # 1024 x 8 x 8
        # self.res_2 = nn.Sequential(
        #     self.convolution_block(1024, 1024), self.convolution_block(1024, 1024)
        # )  # 1024 x 8 x 8
        #
        # self.conv_5 = self.convolution_block(1024, 2048, pool=True)  # 256 x 8 x 8
        # self.conv_6 = self.convolution_block(2048, 4096, pool=True)  # 512 x 4 x 4
        # self.res_3 = nn.Sequential(
        #     self.convolution_block(4096, 4096), self.convolution_block(4096, 4096)
        # )  # 512 x 4 x 4
        #
        # self.output_layer = nn.Sequential(
        #     nn.MaxPool2d(4),  # 9216 x 1 x 1
        #     nn.Flatten(),  # 9216
        #     nn.Linear(9216, output_size),
        # )

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 50 x 50
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 25 x 25
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),  # output :256*25*25
            nn.MaxPool2d(5, 5),  # output: 256 x 5 x 5
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size),
        )

    @staticmethod
    def convolution_block(in_chan, out_chan, pool=False):
        conv_block_layers = [
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        ]
        if pool:
            conv_block_layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*conv_block_layers)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        # out = self.conv_1(x)
        # out = self.conv_2(out)
        # out = self.res_1(out) + out    # Residual Block
        # out = self.conv_3(out)
        # out = self.conv_4(out)
        # out = self.res_2(out) + out    # Residual Block
        # out = self.output_layer(out)
        # return out
        return self.network(x)


    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
