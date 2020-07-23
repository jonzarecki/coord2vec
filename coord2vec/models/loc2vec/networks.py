import torch
from torch import nn
from torchvision import models

from coord2vec.config import PRETRAINED_RESNETS_DIR
from coord2vec.models.loc2vec.loc2vec_config import EMB_SIZE


class Loc2Vec(nn.Module):
    """
    todo add documentation
    """

    def __init__(self):
        super(Loc2Vec, self).__init__()
        resnet_num = 18
        self.model = models.resnet18() if resnet_num == 18 else models.resnet50()
        model_fn = f'{PRETRAINED_RESNETS_DIR}/resnet{resnet_num}.pth'
        checkpoint = torch.load(model_fn)
        self.model.load_state_dict(checkpoint)
        # print(self.model.eval())

        self.model.avgpool == nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        num_ftrs = 18432 if resnet_num == 50 else 4608
        self.model.fc = nn.Linear(num_ftrs, EMB_SIZE)

    def forward(self, x):
        x = self.model(x)
        return x
