import logging
import torch.nn.functional as F
from torch import nn
from utils import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionRecMLP(BaseModel):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, num_outputs, dropout_rate=0.25):
        super(EmotionRecMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.dropout_rate = dropout_rate

        self.num_outputs = num_outputs
        self.dropout = nn.Dropout(self.dropout_rate)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.fc4 = nn.Linear(self.hidden_size_3, self.num_outputs)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
