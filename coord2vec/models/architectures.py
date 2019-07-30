from typing import List

import torch.nn as nn
import torchvision.models as models


##########################
##    Architechtures    ##
##########################

# def rgb_pretrained_resnet50(output_dim: int) -> nn.Module:
#     net = models.resnet50(pretrained=True)
#     return _change_last_layer(net, output_dim)
#
#
# def rgb_resnet50(output_dim: int) -> nn.Module:
#     net = models.resnet50(pretrained=False)
#     return _change_last_layer(net, output_dim)
#
#
# def resnet50(n_channels: int, output_dim: int) -> nn.Module:
#     """
#     create a resnet-18 with first and last layers to fit the coord2vec
#     Args:
#         n_channels: number of OSM channels in the input
#         output_dim: dimension of the output - based on all the losses we want
#
#     Returns:
#         A nn.Module of the resnet
#     """
#     resnet = models.resnet50()
#     return _change_last_layer(_change_first_layer(resnet, n_channels), output_dim)


def resnet18(n_channels: int, output_dim: int) -> nn.Module:
    """
    create a resnet-18 with first and last layers to fit the coord2vec
    Args:
        n_channels: number of OSM channels in the input
        output_dim: dimension of the output - based on all the losses we want

    Returns:
        A nn.Module of the resnet
    """
    resnet = models.resnet18(num_classes = output_dim)
    resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return resnet.float()


def multihead_model(architecture: nn.Module, heads: List[nn.Module]):
    class MultiHeadResnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = architecture
            self.heads = nn.ModuleList(heads)

        def forward(self, x):
            x1 = self.architecture(x)
            outputs = tuple([head(x1) for head in self.heads])
            return x1, outputs

    return MultiHeadResnet()


##########################
##        heads         ##
##########################

def dual_fc_head(input_dim, hidden_dim=128):
    head = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1))
    return head.float()


if __name__ == '__main__':
    print("Zarecki is ugly, and forever will he be")
