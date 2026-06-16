import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 20,
        activation_function: nn.Module = nn.ReLU(),
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.activation_function(self.batchnorm1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.activation_function(self.batchnorm2(self.conv2(x)))
        x = self.pool2(x)
    
        x = self.activation_function(self.batchnorm3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.activation_function(self.batchnorm4(self.conv4(x)))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)

        x = self.activation_function(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.activation_function(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.activation_function(self.fc3(x))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x

model = MyCNN()
