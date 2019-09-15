import torch
import torch.nn as nn
from unittest import TestCase

from coord2vec.models.architectures import resnet18, multihead_model, dual_fc_head, simple_cnn


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

    def test_simpleCNN(self):
        images = torch.ones(10, 3, 224, 224)
        cnn = simple_cnn(3, 128)
        output = cnn.forward(images)
        self.assertTupleEqual(output.shape, (10, 128))

    def test_resnet18(self):
        model = resnet18(10, 10)
        assert (isinstance(model, nn.Module))

    def test_multihead_model(self):
        n_channels = 10
        z_dim = 128
        model = resnet18(n_channels, z_dim)

        head1 = dual_fc_head(z_dim)
        head2 = dual_fc_head(z_dim)

        model = multihead_model(model, [head1, head2])
        assert (isinstance(model, nn.Module))

        data = torch.ones(5, n_channels, 224, 224)
        embedding, output = model.forward(data)
        self.assertEqual(len(output), 2)
