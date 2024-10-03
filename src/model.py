import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights


def get_resnet34(num_classes=81, pretrained=True):
    # Load pre-trained ResNet34 model
#    model = models.resnet34(pretrained=pretrained)
    model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

    # Modify the last fully connected layer for your specific number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

