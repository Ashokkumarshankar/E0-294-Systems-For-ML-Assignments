import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallNet(nn.Module):
    def __init__(self, mnist=True):
      
        super(SmallNet, self).__init__()
        if mnist:
          num_channels = 1
        else:
          num_channels = 3
          
        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.quant = torch.quantization.QuantStub()

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.relu1(self.conv1(x))
        x = self.mp1(x)
        x = self.relu2(self.conv2(x))
        x = self.mp2(x)

        x = x.reshape(-1, 4*4*50)   
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        
        return F.log_softmax(x, dim=1)