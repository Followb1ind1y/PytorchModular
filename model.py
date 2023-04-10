"""
Contains deep learning models for different tasks.
"""

import torch
from torch import nn

__all__ = ["LeNet", "AlexNet", "VGG11",]

class LeNet(nn.Module):
    """
    The LeNet-5 model. (Input shape = d*28*28 -> Output Size = num_classes)

    Args:
        num_classes: The number of output class.

    Example usage:
        model_LeNet = LeNet(num_classes=10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.cov = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                                 nn.Sigmoid(),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                                 nn.Sigmoid(),
                                 nn.AvgPool2d(kernel_size=2, stride=2)
                                 )
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(in_features=16*5*5, out_features=120),
                                nn.Sigmoid(),
                                nn.Linear(in_features=120, out_features=84),
                                nn.Sigmoid(),
                                nn.Linear(in_features=84, out_features=num_classes)
                                )
    
    def forward(self, x):
        cov_out = self.cov(x)
        fc_out = self.fc(cov_out)
        return fc_out

class AlexNet(nn.Module):
    """
    The AlexNet model. (Input shape = d*224*224 -> Output Size = num_classes)

    Args:
        num_classes: The number of output class.

    Example usage:
        model_AlexNet = AlexNet(num_classes=10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.cov = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 )
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(in_features=256*5*5, out_features=4096), nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features=4096, out_features=4096), nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features=4096, out_features=num_classes)
                                )
    
    def forward(self, x):
        cov_out = self.cov(x)
        fc_out = self.fc(cov_out)
        return fc_out

class VGG11(nn.Module):
    """
    The VGG11 model. (Input shape = d*224*224 -> Output Size = num_classes)

    Args:
        arch: A Tuple contains informations of vgg blocks (e.g. arch = ((1, 1, 64), (1, 64, 128))).
        fc_features: The number of input features for the fully-connected layer.
        num_classes: The number of output class.

    Example usage:
        model_VGG11 = VGG11(arch=conv_arch, fc_features=fc_features, num_classes=10)
    """
    def __init__(self, arch, fc_features, num_classes=10):
        super().__init__()
        self.net = nn.Sequential()

        for i, (num_convs, in_channels, out_channels) in enumerate(arch):
            self.net.add_module("vgg_block_" + str(i+1), self.vgg_block(num_convs, in_channels, out_channels))

        self.net.add_module("fc", nn.Sequential(nn.Flatten(),
                                nn.Linear(in_features=fc_features, out_features=4096), nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features=4096, out_features=4096), nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features=4096, out_features=num_classes)
                            ))
    
    def forward(self, x):
        return self.net(x)

    def vgg_block(self, num_convs, in_channels, out_channels):
        """
        Take the information from arch, create the VGG block.

        Args:
            num_convs: Number of convolutional layers in the VGG block.
            in_channels: Number of input channels in the VGG block.
            out_channels: Number of output channels in the VGG block.

        Returns:
            A sequential container contains all the layers in the VGG block.
        """
        layers = []
        for i in range(num_convs):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        return nn.Sequential(*layers)