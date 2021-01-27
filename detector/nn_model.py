"""
Set parameters for neural network model here. Test how many different 
convolutional and linear layers is optimal for your task. Remember the larger
the number of outputs the more ram is needed. Also choose an activation function,
F.relu is standard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) # 1 image, 32 outputs, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # pass through random data to flatten for linear
        x = torch.randn(128, 72).view(-1, 1, 128, 72)
        self._to_linear = None
        self.convs(x)


        self.fc1 = nn.Linear(self._to_linear, 512) # flattening
        self.fc2 = nn.Linear(512, 2) # 2 output classes

    def convs(self, x):
        """Applies activation function on convolutional layers of network"""

        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        # Flatten
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        """Applies activation function on linear layers of network"""

        x = self.convs(x)
        x = x.view(-1, self._to_linear) # .view is reshape, flattens x before
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = Net()
print(net)