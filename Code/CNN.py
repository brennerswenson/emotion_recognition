import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionRecCNN(nn.Module):

    def __init__(self):
        super(EmotionRecCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)