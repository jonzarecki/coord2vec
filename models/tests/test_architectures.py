import torch.nn as nn
from unittest import TestCase

from models.architectures import rgb_pretrained_resnet50, rgb_resnet50, resnet50


class TestRgb_pretrained_resnet50(TestCase):
    def test_rgb_pretrained_resnet50(self):
        model = rgb_pretrained_resnet50()
        assert (isinstance(model, nn.modules.Module))

    def test_rgb_resnet50(self):
        model = rgb_resnet50()
        assert (isinstance(model, nn.modules.Module))

    def test_resnet50(self):
        model = resnet50(10)
        assert (isinstance(model, nn.modules.Module))
