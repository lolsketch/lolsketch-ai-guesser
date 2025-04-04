import torch
import torchvision
from constants import *

# Uncomment this if running on macos
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

def get_model(model_type, pretrained):
    weights = None
    if pretrained:
        weights = 'IMAGENET1K_V1'
        if model_type == 'resnet':
            weights = 'IMAGENET1K_V2'
        elif model_type == 'cnn':
            raise ValueError('No pretrained weights for CNN model')
    
    if model_type == 'cnn':
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Flatten(),
            torch.nn.Linear(128*4*4, num_classes)
        )
    elif model_type == 'vgg':
        model = torchvision.models.vgg16(weights=weights)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_type == 'resnet':
        model = torchvision.models.resnet50(weights=weights)
        model.fc = torch.nn.Linear(2048, num_classes)
    elif model_type == 'mobilenet':
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Linear(576, num_classes)
    else:
        raise ValueError("model_type must be one of: 'cnn', 'vgg', 'resnet', or 'mobilenet'")
    
    return model
    