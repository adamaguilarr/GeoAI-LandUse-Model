import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Input: [3, 64, 64]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # halves H/W each time
        
        # After two poolings: [64, 16, 16]
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [64, 16, 16]
        
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def build_model(num_classes=10):
    return SimpleCNN(num_classes)