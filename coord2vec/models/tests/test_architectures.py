import torch
import torch.nn as nn
from unittest import TestCase

from coord2vec.models.architectures import resnet18, multihead_model, dual_fc_head


class TestArchitectures(TestCase):
    # def test_rgb_pretrained_resnet50(self):
    #     model = rgb_pretrained_resnet50(10)
    #     assert (isinstance(model, nn.Module))
    #
    # def test_rgb_resnet50(self):
    #     model = rgb_resnet50(10)
    #     assert (isinstance(model, nn.Module))
    #
    # def test_resnet50(self):
    #     model = resnet50(10, 10)
    #     assert (isinstance(model, nn.Module))

    def test_resnet18(self):
        model = resnet18(10, 10)
        assert (isinstance(model, nn.Module))

    def test_multihead_resnet18(self):
        n_channels = 10
        z_dim = 128
        model = resnet18(10, z_dim)
        head1 = dual_fc_head(z_dim, n_classes=5)
        head2 = dual_fc_head(z_dim, n_classes=10)

        model = multihead_model(model, [head1, head2])
        assert (isinstance(model, nn.Module))

        data = torch.ones(5, n_channels, 224, 224)
        model.forward(data)
