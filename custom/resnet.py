
import torch.nn as nn
from torch.nn import functional as F


#resnet model

class ResBlock(nn.Module):
    def __init__(self,
                inchannels,
                outchannels,
                kernel_size=3,
                stride=1,
                skip=True):
        super().__init__()
        # Determines whether to add the identity mapping skip connection
        self.skip = skip
        
        # First block of the residual connection
        self.block = nn.Sequential(
            nn.Conv2d(inchannels,
                    outchannels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels,
                    outchannels,
                    kernel_size=kernel_size,
                    padding=1,
                    bias=False),
            nn.BatchNorm2d(outchannels),
        )
        
        # If the stride is 2 or input channels and output channels do not match,
        # then add a convolutional layer and a batch normalization layer to the identity mapping
        if stride == 2 or inchannels != outchannels:
            self.skip = False
            self.skip_conv = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(outchannels)

    def forward(self, x):
        out = self.block(x)
        # If the skip connection is active, add the input to the output
        # If the skip connection is not active, add the skip connection to the output
        if not self.skip:
            out += self.skip_bn(self.skip_conv(x))
        else:
            out += x
        
        out = F.relu(out.clone())
        return out


class ResNet5M(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial convolutional layer and batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        # Residual blocks
        self.resblock3 = ResBlock(64, 64, stride=1)
        self.resblock6 = ResBlock(64, 64, stride=1)
        self.resblock7 = ResBlock(64, 64, stride=1)
        self.resblock8 = ResBlock(64, 128, stride=2)
        self.resblock9 = ResBlock(128, 128, stride=1)
        self.resblock10 = ResBlock(128, 128, stride=1)
        self.resblock11 = ResBlock(128, 128, stride=1)
        self.resblock12 = ResBlock(128, 128, stride=1)
        self.resblock13 = ResBlock(128, 128, stride=1)
        self.resblock14 = ResBlock(128, 512, stride=2)
        
        # Global average pooling and fully-connected layer
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flat = nn.Flatten()
        # self.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x.clone())
        x = self.maxpool(x)
        x = self.resblock3(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
        x = self.resblock14(x)
        x = self.avgpool(x)
        x = self.flat(x)
        # x = self.fc(x) 
        return x
