"""
Contains deep learning models for different tasks.
"""

import torch

from torch import nn
from torch.nn import functional as F

__all__ = ["LeNet", "LeNetBatchNorm", "AlexNet", "VGG11", "NiN", "GoogLeNet", "ResNet18"]

def init_cnn(module):
    """
    Initialize weights for CNNs using Xavier initialization.
    """
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
        
class LeNet(nn.Module):
    """
    The LeNet-5 model. (Input shape = in_c*28*28 -> Output Size = num_classes)

    Args:
        num_classes: The number of output class.

    Example usage:
        model_LeNet = LeNet(in_c=1, num_classes=10)
    """
    def __init__(self, in_c, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=6, kernel_size=5, padding=2),
                                 nn.Sigmoid(),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                                 nn.Sigmoid(),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.Flatten(),
                                 nn.Linear(in_features=16*5*5, out_features=120),
                                 nn.Sigmoid(),
                                 nn.Linear(in_features=120, out_features=84),
                                 nn.Sigmoid(),
                                 nn.Linear(in_features=84, out_features=num_classes)
                                 )
        self.net.apply(init_cnn)
    
    def forward(self, x):
        return self.net(x)

class LeNetBatchNorm(nn.Module):
    """
    The LeNet model with Batch Normalization. (Input shape = in_c*28*28 -> Output Size = num_classes)

    Args:
        num_classes: The number of output class.

    Example usage:
        model_LeNetBatchNorm = LeNetBatchNorm(in_c=1, num_classes=10)
    """
    def __init__(self, in_c, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=6, kernel_size=5, padding=2),
                                 nn.BatchNorm2d(6),
                                 nn.Sigmoid(),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                                 nn.BatchNorm2d(16),
                                 nn.Sigmoid(),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.Flatten(),
                                 nn.Linear(in_features=16*5*5, out_features=120),
                                 nn.BatchNorm1d(120),
                                 nn.Sigmoid(),
                                 nn.Linear(in_features=120, out_features=84),
                                 nn.BatchNorm1d(84),
                                 nn.Sigmoid(),
                                 nn.Linear(in_features=84, out_features=num_classes)
                                 )
    
    def forward(self, x):
        return self.net(x)

class AlexNet(nn.Module):
    """
    The AlexNet model. (Input shape = in_c*224*224 -> Output Size = num_classes)

    Args:
        num_classes: The number of output class.

    Example usage:
        model_AlexNet = AlexNet(in_c=1, num_classes=10)
    """
    def __init__(self, in_c, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Flatten(),
                                 nn.Linear(in_features=256*5*5, out_features=4096), nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=4096, out_features=4096), nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=4096, out_features=num_classes)
                                 )
        self.net.apply(init_cnn)
    
    def forward(self, x):
        return self.net(x)

class VGG11(nn.Module):
    """
    The VGG11 model. (Input shape = in_c*224*224 -> Output Size = num_classes)

    Args:
        arch: A Tuple contains informations of vgg blocks 
                (e.g. arch [num_convs, in_channels, out_channels] = ((1, 1, 64), (1, 64, 128))).
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
        self.net.apply(init_cnn)
    
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

class NiN(nn.Module):
    """
    The Network in Network model. (Input shape = in_c*224*224 -> Output Size = num_classes)

    Args:
        num_classes: The number of output class.

    Example usage:
        model_NiN = NiN(in_c=1, num_classes=10)
    """
    def __init__(self, in_c, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(self.nin_block(in_channels=in_c, out_channels=96, kernel_size=11, stride=4, padding=0),
                                 nn.MaxPool2d(3, stride=2),
                                 self.nin_block(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
                                 nn.MaxPool2d(3, stride=2),
                                 self.nin_block(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
                                 nn.MaxPool2d(3, stride=2),
                                 nn.Dropout(0.5),
                                 self.nin_block(in_channels=384, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten()
                                 )
        self.net.apply(init_cnn)
    
    def forward(self, x):
        return self.net(x)

    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Take the kernel information, create the NiN block.

        Args:
            in_channels: Number of input channels in the NiN block.
            out_channels: Number of output channels in the NiN block.
            kernel_size: The kernel size of the first Conv2d layer.
            stride: The stride of the first Conv2d layer.
            padding: The padding of the first Conv2d layer.

        Returns:
            A sequential container contains three Conv2d layers in the NiN block.
        """
        block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
                              nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                              nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()                           
                              )

        return block

class Inception(nn.Module):
    """
    Initial the Inception Block including four branches,
    concatenate them along the channel dimension at the end

    Args:
        in_c: The number of input channel.
        c1: The output channel number of the b1_1 block.
        c2: A Tuple which contains the output channels number of the b2_1 and b2_2 block.
        c3: A Tuple which contains the output channels number of the b3_1 and b3_2 block.
        c4: The output channel number of the b4_2 block.

    Example usage:
        Inception(in_c=192, c1=64, c2=(96,128), c3=(16,32), c4=32)
    """
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.b1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        self.b2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.b2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.b3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.b3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.Conv2d(in_c, c4, kernel_size=1)
    
    def forward(self,x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)

class GoogLeNet(nn.Module):
    """
    The GoogleNet model. (Input shape = in_c*96*96 -> Output Size = num_classes)

    Args:
        num_classes: The number of output class.

    Example usage:
        model_GoogLeNet = GoogLeNet(in_c=1, num_classes=10)
    """
    def __init__(self, in_c, num_classes=10):
        super().__init__()
        self.in_c= in_c
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),self.b5(), 
                                 nn.Linear(in_features=1024, out_features=num_classes))
        self.net.apply(init_cnn)
    
    def forward(self, x):
        return self.net(x)

    def b1(self):
        return nn.Sequential(nn.Conv2d(in_channels=self.in_c, out_channels=64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b2(self):
        return nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), nn.ReLU(),
                             nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b3(self):
        return nn.Sequential(Inception(in_c=192, c1=64, c2=(96,128), c3=(16,32), c4=32),
                             Inception(in_c=256, c1=128, c2=(128,192), c3=(32,96), c4=64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
    def b4(self):
        return nn.Sequential(Inception(in_c=480, c1=192, c2=(96,208), c3=(16,48), c4=64),
                             Inception(in_c=512, c1=160, c2=(112,224), c3=(24,64), c4=64),
                             Inception(in_c=512, c1=128, c2=(128, 256), c3=(24,64), c4=64),
                             Inception(in_c=512, c1=112, c2=(144,288), c3=(32,64), c4=64),
                             Inception(in_c=528, c1=256, c2=(160,320), c3=(32,128), c4=128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
    def b5(self):
        return nn.Sequential(Inception(in_c=832, c1=256, c2=(160,320), c3=(32,128), c4=128),
                             Inception(in_c=832, c1=384, c2=(192,384), c3=(48,128), c4=128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

class Residual(nn.Module):
    """
    Initial the Residual Block including two Conv2d layers and one optional 1x1 Conv2d layer

    Args:
        in_c: The number of input channel.
        out_c: The number of output channel.
        use_1x1conv: A Boolean to decide if we will use 1x1 Conv to change out channel in this block.
        stride: The stride for the first Conv2d layer.

    Example usage:
        Residual(in_channels, out_channels, use_1x1conv=True, stride=2)
    """
    def __init__(self, in_c, out_c, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
    
    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class ResNet18(nn.Module):
    """
    The ResNet18 model. (Input shape = in_c*224*224 -> Output Size = num_classes)

    Args:
        arch: A Tuple contains informations of residual blocks 
                (e.g. arch [num_residuals, in_channels, out_channels] = ((2, 64, 64), (2, 64, 128))).
        num_classes: The number of output class.

    Example usage:
        model_ResNet18 = ResNet18(arch=((2, 64, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512)), num_classes=10)
    """
    def __init__(self, arch, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('fc', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes)))
        self.net.apply(init_cnn)
    
    def forward(self, x):
        return self.net(x)

    def b1(self):
        return nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                             nn.BatchNorm2d(64), 
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
    def block(self, num_residuals, in_channels, out_channels, first_block=False):
        """
        Take the information from arch, create modules made up of residual blocks using in ResNet18

        Args:
            num_residuals: Number of residual blocks in this Residual Module.
            in_channels: Number of input channels in this Residual Module.
            out_channels: Number of output channels in this Residual Module.

        Returns:
            A sequential container contains all the layers in this Residual Module.
        """
        if first_block:
            assert in_channels == out_channels
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)