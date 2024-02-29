import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM
import torchvision.models as models
import pywt
from PIL import Image
import numpy as np

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        # self.net = models.resnet50(pretrained=True)  # You can use pretrained=False if you don't want pre-trained weights
        # in_features = self.net.fc.in_features
        # self.net.fc = nn.Linear(in_features, 2)  # Assuming 2 classes in the final layer

        # self.net = models.mobilenet_v2(pretrained=True)  # You can use pretrained=False if you don't want pre-trained weights
        # in_features = self.net.classifier[-1].in_features
        # self.net.classifier[-1] = nn.Linear(in_features, 2)

        self.net=EfficientNet.from_pretrained("efficientnet-b5",advprop=True,num_classes=2)


        self.cel = nn.CrossEntropyLoss()
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, x, target):
        for i in range(2):
            pred = self(x)
            if i == 0:
                pred_first = pred

            loss = self.cel(pred, target)

            self.optimizer.zero_grad()
            loss.backward()

            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)

        return pred_first


if __name__ == '__main__':
    from pprint import pprint
    model = Detector()
    # print(model)