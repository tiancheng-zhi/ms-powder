import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from deeplab import DeepLabv3_plus 


class PowderNet(nn.Module):

    def __init__(self, arch, n_channels, n_classes):
        super(PowderNet, self).__init__()
        if arch == 'deeplab':
            self.body = DeepLabv3_plus(nInputChannels=n_channels, n_classes=n_classes, pretrained=False, _print=False)
        else:
            assert(False)

    def forward(self, x):
        out = self.body(x)
        return out


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.body.xception_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.body.aspp1, model.body.aspp2, model.body.aspp3, model.body.aspp4, model.body.conv1, model.body.conv2, model.body.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k
