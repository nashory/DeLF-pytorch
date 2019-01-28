"""
config.py
"""

import argparse
import time
import torchvision.models as models

# helper func.
def str2bool(v):
    return v.lower() in ('true', '1')


# Parser
parser = argparse.ArgumentParser('delf')

# Common options.
parser.add_argument('--gpu_id', 
                    default='4', 
                    type=str, 
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--manualSeed', 
                    type=int, 
                    default=int(time.time()), 
                    help='manual seed')
# Experiment
parser.add_argument('--expr', 
                    default='devel', 
                    type=str, 
                    help='experiment name')
parser.add_argument('--load_from', 
                    default='dummy',
                    type=str, 
                    help='from which experiment the model be loaded')
# Datasets
parser.add_argument('--stage', 
                    default='finetune', 
                    type=str, 
                    help='target stage: finetune | keypoint')
parser.add_argument('--train_path_for_pretraining', 
                    default='../../data/landmarks/landmarks_full_train', 
                    type=str)
parser.add_argument('--val_path_for_pretraining', 
                    default='../../data/landmarks/landmarks_full_val', 
                    type=str)
parser.add_argument('--train_path_for_finetuning', 
                    default='../../data/landmarks/landmarks_clean_train', 
                    type=str)
parser.add_argument('--val_path_for_finetuning', 
                    default='../../data/landmarks/landmarks_clean_val', 
                    type=str)
parser.add_argument('--workers', 
                    default=20, 
                    type=int,
                    help='number of data loading workers (default: 4)')
# preprocessing
parser.add_argument('--finetune_sample_size',
                    default=256,
                    type=int,
                    help='finetune resize (default: 256)')
parser.add_argument('--finetune_crop_size',
                    default=224,
                    type=int,
                    help='finetune crop (default: 224)')
parser.add_argument('--keypoint_sample_size', 
                    default=900, 
                    type=int,
                    help='keypoint (default: 900)')
parser.add_argument('--keypoint_crop_size', 
                    default=720, 
                    type=int,
                    help='keypoint (default: 720)')
parser.add_argument('--target_layer', 
                    default='layer3', 
                    type=str,
                    help='target layer you wish to extract local features from: layer3 | layer4')
parser.add_argument('--use_random_gamma_rescale',
                    type=str2bool,
                    default=True,
                    help='apply gamma rescaling in range of [0.3535, 1.0]')
# training parameters
parser.add_argument('--finetune_epoch',
                    default=30,
                    type=int,
                    help='number of total epochs for finetune stage.')
parser.add_argument('--keypoint_epoch',
                    default=30,
                    type=int,
                    help='number of total epochs for keypoint stage.')
parser.add_argument('--lr',
                    default=0.008,
                    type=float,
                    help='initial learning rate')
parser.add_argument('--lr_gamma',
                    default=0.5,
                    type=float,
                    help='decay factor of learning rate')
parser.add_argument('--lr_stepsize',
                    default=10,
                    type=int,
                    help='decay learning rate at every specified epoch.')
parser.add_argument('--weight_decay',
                    default=0.0001,
                    type=float,
                    help='weight decay (l2 penalty)')
parser.add_argument('--optim',
                    default='sgd',
                    type=str,
                    help='optimizer: rmsprop | sgd | adam')
parser.add_argument('--train_batch_size',
                    default=8,
                    type=int,
                    help='train batchsize (default: 16)')
parser.add_argument('--val_batch_size',
                    default=8,
                    type=int,
                    help='val batchsize (default: 16)')
parser.add_argument('--ncls',
                    default=586,
                    type=int,
                    help='number of classes')
parser.add_argument('--lr_decay',
                    default=0.5,
                    type=float,
                    help='lr decay factor')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet50',
                    choices=['resnet50, resnet101, resnet152'],
                    help='only support resnet50 at the moment.')

## parse and save config.
config, _ = parser.parse_known_args()


