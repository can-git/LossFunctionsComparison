import torch
import torch.nn as nn
import torch.nn.functional as F
import CFG as p
from torchvision.models.resnet import resnet50


# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define 2D convolutional layers
        self.backbone = resnet50(pretrained=True)
        # Replace the last fully connected layer with a new one that outputs num_classes scores
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, p.NUM_CLASSES)

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        return x
