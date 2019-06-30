import torch.nn as nn
from unittest import TestCase

from models.architectures import rgb_pretrained_resnet50, rgb_resnet50, resnet50, resnet18


class TestArchitectures(TestCase):
    def test_rgb_pretrained_resnet50(self):
        model = rgb_pretrained_resnet50(10)
        assert (isinstance(model, nn.Module))

    def test_rgb_resnet50(self):
        model = rgb_resnet50(10)
        assert (isinstance(model, nn.Module))

    def test_resnet50(self):
        model = resnet50(10, 10)
        assert (isinstance(model, nn.Module))

    def test_resnet18(self):
        model = resnet18(10, 10)
        assert (isinstance(model, nn.Module))
