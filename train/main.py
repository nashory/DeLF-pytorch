'''
main.py
'''

import os, sys, time
sys.path.append('../')
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from config import config


def main():
    # print config.
    state = {k: v for k, v in config._get_kwargs()}
    print(state)

    # if use cuda.
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    use_cuda = torch.cuda.is_available()

    # Random seed
    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(config.manualSeed)
        torch.backends.cudnn.benchmark = True           # speed up training.
    
    # data loader
    from dataloader import get_loader
    if config.stage in ['finetune']:
        sample_size = config.finetune_sample_size
        crop_size = config.finetune_crop_size
    elif config.stage in ['keypoint']:
        sample_size = config.keypoint_sample_size
        crop_size = config.keypoint_crop_size
   
    # dataloader for pretrain
    train_loader_pt, val_loader_pt = get_loader(
        train_path = config.train_path_for_pretraining,
        val_path = config.val_path_for_pretraining,
        stage = config.stage,
        train_batch_size = config.train_batch_size,
        val_batch_size = config.val_batch_size,
        sample_size = sample_size,
        crop_size = crop_size,
        workers = config.workers)
    # dataloader for finetune
    train_loader_ft, val_loader_ft = get_loader(
        train_path = config.train_path_for_finetuning,
        val_path = config.val_path_for_finetuning,
        stage = config.stage,
        train_batch_size = config.train_batch_size,
        val_batch_size = config.val_batch_size,
        sample_size = sample_size,
        crop_size = crop_size,
        workers = config.workers)
    

    # load model
    from delf import Delf_V1
    model = Delf_V1(
        ncls = config.ncls,
        load_from = config.load_from,
        arch = config.arch,
        stage = config.stage,
        target_layer = config.target_layer,
        use_random_gamma_rescale = config.use_random_gamma_rescale)

    # solver
    from solver import Solver
    solver = Solver(config=config, model=model)
    if config.stage in ['finetune']:
        epochs = config.finetune_epoch
    elif config.stage in ['keypoint']:
        epochs = config.keypoint_epoch

    # train/test for N-epochs. (50%: pretain with datasetA, 50%: finetune with datasetB)
    for epoch in range(epochs):
        if epoch < int(epochs * 0.5):
            print('[{:.1f}] load pretrain dataset: {}'.format(
                float(epoch) / epochs,
                config.train_path_for_pretraining))
            train_loader = train_loader_pt
            val_loader = val_loader_pt
        else:
            print('[{:.1f}] load finetune dataset: {}'.format(
                float(epoch) / epochs,
                config.train_path_for_finetuning))
            train_loader = train_loader_ft
            val_loader = val_loader_ft

        solver.train('train', epoch, train_loader, val_loader)
        solver.train('val', epoch, train_loader, val_loader)

    print('Congrats! You just finished DeLF training.')


if __name__ == '__main__':
    main()
