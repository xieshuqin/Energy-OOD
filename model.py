import torch
import torch.nn as nn
import torchvision


def build_model():
    return EnergyWideResNet()


class EnergyWideResNet(nn.Module):
    def __init__(self):
        super(EnergyWideResNet, self).__init__()
        self.model = torchvision.models.wide_resnet50_2(pretrained=True)
        self.model.fc = nn.Linear(512 * 4, 10)

        # # Only finetune last layer
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        return self.model(x)
