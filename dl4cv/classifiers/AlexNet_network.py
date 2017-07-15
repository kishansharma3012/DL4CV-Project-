import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

class ClassificationNetwork(nn.Module):

    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        ############################################################################
        #                             YOUR CODE
        #                      #
        ############################################################################
        self.alex_model = models.alexnet(pretrained = True)
        
        for param in self.alex_model.parameters():
            param.requires_grad = False

        #for param in self.alex_model.features[8].parameters():
        #    param.requires_grad = True

        for param in self.alex_model.features[10].parameters():
            param.requires_grad = True
        
        #self.alex_conv = nn.Sequential(
        #                alex_model.features,
        #                alex_model.classifier,
        #                )

        self.my_model = nn.Sequential(
                        nn.Linear(1000, 400, bias=True),
                        nn.Linear(400, 38, bias=True),
                        )
                        
        for param in self.my_model.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ############################################################################
        #                             YOUR CODE                                    #
        ############################################################################
        #out = self.alex_conv(x)
        out = self.alex_model.forward(x)
        out = self.my_model(out)
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
