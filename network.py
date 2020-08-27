# import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, feature=2, output=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # 32x32x3 -> 28x28x6 (-> 14x14x6)
        self.conv2 = nn.Conv2d(6, 16, 5)  # 14x14x6 -> 10x10x16 (-> 5x5x16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
