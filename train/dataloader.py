
#-*- coding: utf-8 -*-

'''
dataloader.py
'''

import sys, os, time

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 1000000000     # to avoid error "https://github.com/zimeon/iiif/issues/11"
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid error "https://github.com/python-pillow/Pillow/issues/1510"

def get_loader(
    train_path,
    val_path,
    stage,
    train_batch_size,
    val_batch_size,
    sample_size,
    crop_size,
    workers):

    if stage in ['finetune']:
        # for train
        prepro = []
        prepro.append(transforms.Resize(size=sample_size))
        prepro.append(transforms.CenterCrop(size=sample_size))
        prepro.append(transforms.RandomCrop(size=crop_size, padding=0))
        prepro.append(transforms.RandomHorizontalFlip())
        #prepro.append(transforms.RandomRotation((-15, 15)))        # experimental.
        prepro.append(transforms.ToTensor())
        train_transform = transforms.Compose(prepro)
        train_path = train_path
        
        # for val
        prepro = []
        prepro.append(transforms.Resize(size=sample_size))
        prepro.append(transforms.CenterCrop(size=crop_size))
        prepro.append(transforms.ToTensor())
        val_transform = transforms.Compose(prepro)
        val_path = val_path

    elif stage in ['keypoint']:
        # for train
        prepro = []
        prepro.append(transforms.Resize(size=sample_size))
        prepro.append(transforms.CenterCrop(size=sample_size))
        prepro.append(transforms.RandomCrop(size=crop_size, padding=0))
        prepro.append(transforms.RandomHorizontalFlip())
        #prepro.append(transforms.RandomRotation((-15, 15)))        # experimental.
        prepro.append(transforms.ToTensor())
        train_transform = transforms.Compose(prepro)
        train_path = train_path
        
        # for val
        prepro = []
        prepro.append(transforms.Resize(size=sample_size))
        prepro.append(transforms.CenterCrop(size=crop_size))
        prepro.append(transforms.ToTensor())
        val_transform = transforms.Compose(prepro)
        val_path = val_path
    
    # image folder dataset.
    train_dataset = datasets.ImageFolder(root = train_path,
                                         transform = train_transform)
    val_dataset = datasets.ImageFolder(root = val_path,
                                       transform = val_transform)

    # return train/val dataloader
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = train_batch_size,
                                               shuffle = True,
                                               num_workers = workers)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                             batch_size = val_batch_size,
                                             shuffle = False,
                                             num_workers = workers)

    return train_loader, val_loader



