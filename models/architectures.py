import torch.nn as nn
import torchvision.models as models

def rgb_pretrained_resnet50():
    return models.resnet50(pretrained=True)

def rgb_resnet50():
    return models.resnet50()

def resnet50(n_channels):
    resnet = models.resnet50()
    resnet_except_first = nn.Sequential(*list(resnet.children())[1:])
    first_conv = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return nn.Sequential(first_conv, resnet_except_first)

if __name__ == '__main__':
    print("Zarecki is ugly")