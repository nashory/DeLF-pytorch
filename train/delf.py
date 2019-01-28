
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, time
sys.path.append('../')
import random
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from train.layers import (
    CMul, 
    Flatten, 
    ConcatTable, 
    Identity, 
    Reshape, 
    SpatialAttention2d, 
    WeightedSum2d)


''' helper functions
'''

def __unfreeze_weights__(module_dict, freeze=[]):
    for _, v in enumerate(freeze):
        module = module_dict[v]
        for param in module.parameters():
            param.requires_grad = True
    
def __freeze_weights__(module_dict, freeze=[]):
    for _, v in enumerate(freeze):
        module = module_dict[v]
        for param in module.parameters():
            param.requires_grad = False

def __print_freeze_status__(model):
    '''print freeze stagus. only for debugging purpose.
    '''
    for i, module in enumerate(model.named_children()):
        for param in module[1].parameters():
            print('{}:{}'.format(module[0], str(param.requires_grad)))

def __load_weights_from__(module_dict, load_dict, modulenames):
    for modulename in modulenames:
        module = module_dict[modulename]
        print('loaded weights from module "{}" ...'.format(modulename))
        module.load_state_dict(load_dict[modulename])

def __deep_copy_module__(module, exclude=[]):
    modules = {}
    for name, m in module.named_children():
        if name not in exclude:
            modules[name] = copy.deepcopy(m)
            print('deep copied weights from layer "{}" ...'.format(name))
    return modules

def __cuda__(model):
    if torch.cuda.is_available():
        model.cuda()
    return model


'''Delf
'''

class Delf_V1(nn.Module):
    def __init__(
        self,
        ncls=None,
        load_from=None,
        arch='resnet50',
        stage='inference',
        target_layer='layer3',
        use_random_gamma_rescale=False):

        super(Delf_V1, self).__init__()

        self.arch = arch
        self.stage = stage
        self.target_layer = target_layer
        self.load_from = load_from
        self.use_random_gamma_rescale = use_random_gamma_rescale

        self.module_list = nn.ModuleList()
        self.module_dict = {}
        self.end_points = {}

        in_c = self.__get_in_c__()
        if self.stage in ['finetune']:
            use_pretrained_base = True
            exclude = ['avgpool', 'fc']

        elif self.stage in ['keypoint']:
            use_pretrained_base = False
            self.use_l2_normalized_feature = True
            if self.target_layer in ['layer3']:
                exclude = ['layer4', 'avgpool', 'fc']
            if self.target_layer in ['layer4']:
                exclude = ['avgpool', 'fc']

        else:
            assert self.stage in ['inference']
            use_pretrained_base = False
            self.use_l2_normalized_feature = True
            if self.target_layer in ['layer3']:
                exclude = ['layer4', 'avgpool', 'fc']
            if self.target_layer in ['layer4']:
                exclude = ['avgpool', 'fc']

        if self.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            print('[{}] loading {} pretrained ImageNet weights ... It may take few seconds...'
                    .format(self.stage, self.arch))
            module = models.__dict__[self.arch](pretrained=use_pretrained_base)
            module_state_dict = __deep_copy_module__(module, exclude=exclude)
            module = None

            # endpoint: base
            submodules = []
            submodules.append(module_state_dict['conv1'])
            submodules.append(module_state_dict['bn1'])
            submodules.append(module_state_dict['relu'])
            submodules.append(module_state_dict['maxpool'])
            submodules.append(module_state_dict['layer1'])
            submodules.append(module_state_dict['layer2'])
            submodules.append(module_state_dict['layer3'])
            self.__register_module__('base', submodules)

            # build structure.
            if self.stage in ['finetune']:
                # endpoint: layer4, pool
                self.__register_module__('layer4', module_state_dict['layer4'])
                self.__register_module__('pool', nn.AvgPool2d(
                    kernel_size=7, stride=1, padding=0,
                    ceil_mode=False, count_include_pad=True))
            elif self.stage in ['keypoint', 'inference']:
                # endpoint: attn, pool
                self.__register_module__('attn', SpatialAttention2d(in_c=in_c, act_fn='relu'))
                self.__register_module__('pool', WeightedSum2d())


            if self.stage not in ['inference']:
                # endpoint: logit
                submodules = []
                submodules.append(nn.Conv2d(in_c, ncls, 1))
                submodules.append(Flatten())
                self.__register_module__('logits', submodules)

            # load weights.
            if self.stage in ['keypoint']:
                load_dict = torch.load(self.load_from)
                __load_weights_from__(self.module_dict, load_dict, modulenames=['base'])
                __freeze_weights__(self.module_dict, freeze=['base'])
                print('load model from "{}"'.format(load_from))
            elif self.stage in ['inference']:
                load_dict = torch.load(self.load_from)
                __load_weights_from__(self.module_dict, load_dict, modulenames=['base','attn','pool'])
                print('load model from "{}"'.format(load_from))
                

    def __register_module__(self, modulename, module):
        if isinstance(module, list) or isinstance(module, tuple):
            module = nn.Sequential(*module)
        self.module_list.append(module)
        self.module_dict[modulename] = module

    def __get_in_c__(self):
        # adjust input channels according to arch.
        if self.arch in ['resnet18', 'resnet34']:
            in_c = 512
        elif self.arch in ['resnet50', 'resnet101', 'resnet152']:
            if self.stage in ['finetune']:
                in_c = 2048
            elif self.stage in ['keypoint', 'inference']:
                if self.target_layer in ['layer3']:
                    in_c = 1024
                elif self.target_layer in ['layer4']:
                    in_c = 2048
        return in_c

    def __forward_and_save__(self, x, modulename):
        module = self.module_dict[modulename]
        x = module(x)
        self.end_points[modulename] = x
        return x

    def __forward_and_save_feature__(self, x, model, name):
        x = model(x)
        self.end_points[name] = x.data
        return x

    def __gamma_rescale__(self, x, min_scale=0.3535, max_scale=1.0):
        '''max_scale > 1.0 may cause training failure.
        '''
        h, w = x.size(2), x.size(3)
        assert w == h, 'input must be square image.'
        gamma = random.uniform(min_scale, max_scale)
        new_h, new_w = int(h*gamma), int(w*gamma)
        x = F.upsample(x, size=(new_h, new_w), mode='bilinear')
        return x

    def get_endpoints(self):
        return self.end_points

    def get_feature_at(self, modulename):
        return copy.deepcopy(self.end_points[modulename].data.cpu())

    def write_to(self, state):
        if self.stage in ['finetune']:
            state['base'] = self.module_dict['base'].state_dict()
            state['layer4'] = self.module_dict['layer4'].state_dict()
            state['pool'] = self.module_dict['pool'].state_dict()
            state['logits'] = self.module_dict['logits'].state_dict()
        elif self.stage in ['keypoint']:
            state['base'] = self.module_dict['base'].state_dict()
            state['attn'] = self.module_dict['attn'].state_dict()
            state['pool'] = self.module_dict['pool'].state_dict()
            state['logits'] = self.module_dict['logits'].state_dict()
        else:
            assert self.stage in ['inference']
            raise ValueError('inference does not support model saving!')

    def forward_for_serving(self, x):
        '''
        This function directly returns attention score and raw features
        without saving to endpoint dict.
        '''
        x = self.__forward_and_save__(x, 'base')
        if self.target_layer in ['layer4']:
            x = self.__forward_and_save__(x, 'layer4')
        ret_x = x
        if self.use_l2_normalized_feature:
            attn_x = F.normalize(x, p=2, dim=1)
        else:
            attn_x = x
        attn_score = self.__forward_and_save__(x, 'attn')
        ret_s = attn_score
        return ret_x.data.cpu(), ret_s.data.cpu()

    def forward(self, x):
        if self.stage in ['finetune']:
            x = self.__forward_and_save__(x, 'base')
            x = self.__forward_and_save__(x, 'layer4')
            x = self.__forward_and_save__(x, 'pool')
            x = self.__forward_and_save__(x, 'logits')
        elif self.stage in ['keypoint']:
            if self.use_random_gamma_rescale:
                x = self.__gamma_rescale__(x)
            x = self.__forward_and_save__(x, 'base')
            if self.target_layer in ['layer4']:
                x = self.__forward_and_save__(x, 'layer4')
            if self.use_l2_normalized_feature:
                attn_x = F.normalize(x, p=2, dim=1)
            else:
                attn_x = x
            attn_score = self.__forward_and_save__(x, 'attn')
            x = self.__forward_and_save__([attn_x, attn_score], 'pool')
            x = self.__forward_and_save__(x, 'logits')
        
        elif self.stage in ['inference']:
            x = self.__forward_and_save__(x, 'base')
            if self.target_layer in ['layer4']:
                x = self.__forward_and_save__(x, 'layer4')
            if self.use_l2_normalized_feature:
                attn_x = F.normalize(x, p=2, dim=1)
            else:
                attn_x = x
            attn_score = self.__forward_and_save__(x, 'attn')
            x = self.__forward_and_save__([attn_x, attn_score], 'pool')

        else:
            raise ValueError('unsupported stage parameter: {}'.format(self.stage))
        return x

if __name__=="__main__":
    pass;









