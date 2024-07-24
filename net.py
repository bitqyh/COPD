import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def resnet18(pretrained=False, progress=True, num_classes=4, dropout_rate=0.5):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    # nn.Linear(num_ftrs, num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

if __name__=='__main__':
    in_data=torch.ones(1,3,224,224)
    net= resnet18(pretrained=False, progress=True, num_classes=4)
    out=net(in_data)
    print(out)