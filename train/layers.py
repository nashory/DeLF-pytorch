# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F



'''custom layers
'''
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
    def __repr__(self):
        return self.__class__.__name__


class ConcatTable(nn.Module):
    '''ConcatTable container in Torch7.
    '''
    def __init__(self, layer1, layer2):
        super(ConcatTable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        
    def forward(self,x):
        return [self.layer1(x), self.layer2(x)]


class Identity(nn.Module):
    '''
    nn.Identity in Torch7.
    '''
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    def __repr__(self):
        return self.__class__.__name__ + ' (skip connection)'


class Reshape(nn.Module):
    '''
    nn.Reshape in Torch7.
    '''
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(self.shape)
    def __repr__(self):
        return self.__class__.__name__ + ' (reshape to size: {})'.format(" ".join(str(x) for x in self.shape))


class CMul(nn.Module):
    '''
    nn.CMul in Torch7.
    '''
    def __init__(self):
        super(CMul, self).__init__()
    def forward(self, x):
        return x[0]*x[1]
    def __repr__(self):
        return self.__class__.__name__


class WeightedSum2d(nn.Module):
    def __init__(self):
        super(WeightedSum2d, self).__init__()
    def forward(self, x):
        x, weights = x
        assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3),\
                'err: h, w of tensors x({}) and weights({}) must be the same.'\
                .format(x.size, weights.size)
        y = x * weights                                       # element-wise multiplication
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))      # b x c x hw
        return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # b x c x 1 x 1
    def __repr__(self):
        return self.__class__.__name__


class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    <!!!> attention score normalization will be added for experiment.
    '''
    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)                 # 1x1 conv
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)                    # 1x1 conv
        self.softplus = nn.Softplus(beta=1, threshold=20)       # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        s : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.softplus(x)
        return x
    
    def __repr__(self):
        return self.__class__.__name__

