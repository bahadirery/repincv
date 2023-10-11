import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models


import torchvision
from torchvision.models.resnet import BasicBlock
import warnings

from models import  resnet

views_allowed=['LCC','LMLO','RCC','RMLO']

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

class SILmodel(nn.Module):
    def __init__(self, config_params):
        super(SILmodel, self).__init__()
        self.activation = config_params['activation']
        self.featureextractormodel = config_params['femodel']
        self.extra = config_params['extra']
        self.topkpatch = config_params['topkpatch']
        self.pretrained = config_params['pretrained']
        self.channel = config_params['channel']
        self.regionpooling = config_params['regionpooling']
        self.learningtype = config_params['learningtype']
        
        if self.featureextractormodel:
            #print(self.featureextractormodel)
            
            if self.featureextractormodel == 'resnet18':
                self.feature_extractor = resnet.resnet18(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling, learningtype = self.learningtype)
            elif self.featureextractormodel == 'resnet34':
                self.feature_extractor = resnet.resnet34(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling, learningtype = self.learningtype)
            elif self.featureextractormodel == 'resnet50':
                self.feature_extractor = resnet.resnet50(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling, learningtype = self.learningtype)
            elif self.featureextractormodel == 'densenet121':
                self.feature_extractor = densenet.densenet121(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif self.featureextractormodel == 'densenet169':
                self.feature_extractor = densenet.densenet169(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif self.featureextractormodel == 'convnext-T':
                self.feature_extractor = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                self.feature_extractor.classifier[2] = nn.Linear(768, 1)
            elif self.featureextractormodel == 'gmic_resnet18':
                self.feature_extractor = gmic._gmic(config_params['gmic_parameters'])
    
    def forward(self, x, eval_mode):
        if self.featureextractormodel=='gmic_resnet18':
            y_local, y_global, y_fusion, saliency_map, patch_locations, patches, patch_attns, h_crops = self.feature_extractor(x, eval_mode)
            return y_local, y_global, y_fusion, saliency_map, patch_locations, patches, patch_attns, h_crops
        else:
            M = self.feature_extractor(x)
            M = M.view(M.shape[0],-1)
            print(M.shape)
            return M