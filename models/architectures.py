import torch.nn as nn
import torchvision.models as models


##########################
##    Architechtures    ##
##########################

def rgb_pretrained_resnet50(output_dim: int) -> nn.Module:
    net = models.resnet50(pretrained=True)
    return _change_last_layer(net, output_dim)


def rgb_resnet50(output_dim: int) -> nn.Module:
    net = models.resnet50(pretrained=False)
    return _change_last_layer(net, output_dim)


def resnet50(n_channels: int, output_dim: int) -> nn.Module:
    """
    create a resnet-18 with first and last layers to fit the coord2vec
    Args:
        n_channels: number of OSM channels in the input
        output_dim: dimension of the output - based on all the losses we want

    Returns:
        A nn.Module of the resnet
    """
    resnet = models.resnet50()
    return _change_last_layer(_change_first_layer(resnet, n_channels), output_dim)


def resnet18(n_channels: int, output_dim: int) -> nn.Module:
    """
    create a resnet-18 with first and last layers to fit the coord2vec
    Args:
        n_channels: number of OSM channels in the input
        output_dim: dimension of the output - based on all the losses we want

    Returns:
        A nn.Module of the resnet
    """
    resnet = models.resnet18()
    return _change_last_layer(_change_first_layer(resnet, n_channels), output_dim)


##########################
##    util functions    ##
##########################

def _change_last_layer(net, output_dim):
    list(net.children())[-1].out_features = output_dim
    return net


def _change_first_layer(net, in_channels):
    list(net.children())[0].in_channels = in_channels
    return net


if __name__ == '__main__':
    print("Zarecki is ugly")
