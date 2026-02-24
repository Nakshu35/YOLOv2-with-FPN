import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class Darknet19(nn.Module):
    def __init__(self):
            super().__init__()

            # Layer-1
            self.layer1 = nn.Sequential(
                ConvBlock(3, 32, 3, 1, 1),
                nn.MaxPool2d(2, 2)
            )

            # Layer-2
            self.layer2 = nn.Sequential(
                ConvBlock(32, 64, 3, 1, 1),
                nn.MaxPool2d(2, 2)
            )

            # Layer-3
            self.layer3 = nn.Sequential(
                ConvBlock(64, 128, 3, 1, 1),
                ConvBlock(128, 64, 1, 1, 0),
                ConvBlock(64, 128, 3, 1, 1),
                nn.MaxPool2d(2, 2)
            )

            # Layer-4
            self.layer4 = nn.Sequential(
                ConvBlock(128, 256, 3, 1, 1),
                ConvBlock(256, 128, 1, 1, 0),
                ConvBlock(128, 256, 3, 1, 1),
                nn.MaxPool2d(2, 2)
            )

            # Layer-5
            self.layer5 = nn.Sequential(
                ConvBlock(256, 512, 3, 1, 1),
                ConvBlock(512, 256, 1, 1, 0),
                ConvBlock(256, 512, 3, 1, 1),
                ConvBlock(512, 256, 1, 1, 0),
                ConvBlock(256, 512, 3, 1, 1),
                nn.MaxPool2d(2, 2)
            )

            # Layer-6
            self.layer6 = nn.Sequential(
                ConvBlock(512, 1024, 3, 1, 1),
                ConvBlock(1024, 512, 1, 1, 0),
                ConvBlock(512, 1024, 3, 1, 1),
                ConvBlock(1024, 512, 1, 1, 0),
                ConvBlock(512, 1024, 3, 1, 1)
            )

    def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)

            return x
        