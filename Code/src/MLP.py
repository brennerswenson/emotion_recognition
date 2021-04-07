import logging

import torch.nn.functional as F
from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionRecMLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_outputs):
        super(EmotionRecMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        self.num_outputs = num_outputs
        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.num_outputs)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
