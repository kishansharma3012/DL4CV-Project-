import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

class ClassificationNetwork(nn.Module):

    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        self.resnet18_model = models.resnet18(pretrained = True)
        
        for param in self.resnet18_model.parameters():
            param.requires_grad = True
	
        self.resnet18_model.fc = nn.Linear(512, 38)
     
        
    def forward(self, x):
        out = self.resnet18_model.forward(x)
        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model... %s' % path
        torch.save(self, path)

