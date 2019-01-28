#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PyTorch Implementation of training DeLF feature.
Solver for step 1 (finetune local descriptor)
nashory, 2018.04
'''
import os, sys, time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils import Bar, Logger, AverageMeter, compute_precision_top_k, mkdir_p
    
'''helper functions.
'''
def __cuda__(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

def __is_cuda__():
    return torch.cuda.is_available()

def __to_var__(x, volatile=False):
    return Variable(x, volatile=volatile)

def __to_tensor__(x):
    return x.data


class Solver(object):
    def __init__(self, config, model):
        self.state = {k: v for k, v in config._get_kwargs()} 
        self.config = config
        self.epoch = 0          # global epoch.
        self.best_acc = 0       # global best accuracy.
        self.prefix = os.path.join('repo', config.expr)
        
        # ship model to cuda
        self.model = __cuda__(model)

        # define criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        if config.optim.lower() in ['rmsprop']:
            self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=config.lr,
                                           weight_decay=config.weight_decay)
        elif config.optim.lower() in ['sgd']:
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                       lr=config.lr,
                                       weight_decay=config.weight_decay)
        elif config.optim.lower() in ['adam']:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=config.lr,
                                        weight_decay=config.weight_decay)
        
        # decay learning rate by a factor of 0.5 every 10 epochs
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_stepsize, 
            gamma=config.lr_gamma)

        # create directory to save result if not exist.
        self.ckpt_path = os.path.join(self.prefix, config.stage, 'ckpt')
        self.log_path = os.path.join(self.prefix, config.stage, 'log')
        self.image_path = os.path.join(self.prefix, config.stage, 'image')
        mkdir_p(self.ckpt_path)
        mkdir_p(self.log_path)
        mkdir_p(self.image_path)

        # set logger.
        self.logger = {}
        self.title = 'DeLF-{}'.format(config.stage.upper())
        self.logger['train'] = Logger(os.path.join(self.prefix, config.stage, 'log/train.log'))
        self.logger['val'] = Logger(os.path.join(self.prefix, config.stage, 'log/val.log'))
        self.logger['train'].set_names(
            ['epoch','lr', 'loss', 'top1_accu', 'top3_accu', 'top5_accu'])
        self.logger['val'].set_names(
            ['epoch','lr', 'loss', 'top1_accu', 'top3_accu', 'top5_accu'])
        
    def __exit__(self):
        self.train_logger.close()
        self.val_logger.close()


    def __adjust_pixel_range__(self, 
                             x,
                             range_from=[0,1],
                             range_to=[-1,1]):
        '''
        adjust pixel range from <range_from> to <range_to>.
        '''
        if not range_from == range_to:
            scale = float(range_to[1]-range_to[0])/float(range_from[1]-range_from[0])
            bias = range_to[0]-range_from[0]*scale
            x = x.mul(scale).add(bias)
            return x

    def __save_checkpoint__(self, state, ckpt='ckpt', filename='checkpoint.pth.tar'):
        filepath = os.path.join(ckpt, filename)
        torch.save(state, filepath)
    
    def __solve__(self, mode, epoch, dataloader):
        '''solve
        mode: train / val
        '''
        batch_timer = AverageMeter()
        data_timer = AverageMeter()
        prec_losses = AverageMeter()
        prec_top1 = AverageMeter()
        prec_top3 = AverageMeter()
        prec_top5 = AverageMeter()
        
        if mode in ['val']:
            pass;
            #confusion_matrix = ConusionMeter()
        
        since = time.time()
        bar = Bar('[{}]{}'.format(mode.upper(), self.title), max=len(dataloader))
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # measure data loading time
            data_timer.update(time.time() - since)
            
            # wrap inputs in variable
            if mode in ['train']:
                if __is_cuda__():
                    inputs = inputs.cuda()
                    labels = labels.cuda(async=True)
                inputs = __to_var__(inputs)
                labels = __to_var__(labels)
            elif mode in ['val']:
                if __is_cuda__():
                    inputs = inputs.cuda()
                    labels = labels.cuda(async=True)
                inputs = __to_var__(inputs, volatile=True)
                labels = __to_var__(labels, volatile=False)
            
            # forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # backward + optimize
            if mode in ['train']:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # statistics
            prec_1, prec_3, prec_5 = compute_precision_top_k(
                __to_tensor__(outputs),
                __to_tensor__(labels),
                top_k=(1,3,5))
            batch_size = inputs.size(0)
            prec_losses.update(__to_tensor__(loss)[0], batch_size)
            prec_top1.update(prec_1[0], batch_size)
            prec_top3.update(prec_3[0], batch_size)
            prec_top5.update(prec_5[0], batch_size)
            
            # measure elapsed time
            batch_timer.update(time.time() - since)
            since = time.time()
            
            # progress
            log_msg = ('\n[{mode}][epoch:{epoch}][iter:({batch}/{size})]'+
                        '[lr:{lr}] loss: {loss:.4f} | top1: {top1:.4f} | ' +
                        'top3: {top3:.4f} | top5: {top5:.4f} | eta: ' +
                        '(data:{dt:.3f}s),(batch:{bt:.3f}s),(total:{tt:})') \
                        .format(
                            mode=mode,
                            epoch=self.epoch+1,
                            batch=batch_idx+1,
                            size=len(dataloader),
                            lr=self.lr_scheduler.get_lr()[0],
                            loss=prec_losses.avg,
                            top1=prec_top1.avg,
                            top3=prec_top3.avg,
                            top5=prec_top5.avg,
                            dt=data_timer.val,
                            bt=batch_timer.val,
                            tt=bar.elapsed_td)
            print(log_msg)
            bar.next()
        bar.finish()

        # write to logger
        self.logger[mode].append([self.epoch+1,
                                  self.lr_scheduler.get_lr()[0],
                                  prec_losses.avg,
                                  prec_top1.avg,
                                  prec_top3.avg,
                                  prec_top5.avg])
        
        # save model
        if mode == 'val' and prec_top1.avg > self.best_acc:
            print('best_acc={}, new_best_acc={}'.format(self.best_acc, prec_top1.avg))
            self.best_acc = prec_top1.avg
            state = {
                'epoch': self.epoch,
                'acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
            }
            self.model.write_to(state)
            filename = 'bestshot.pth.tar'
            self.__save_checkpoint__(state, ckpt=self.ckpt_path, filename=filename)


    def train(self, mode, epoch, train_loader, val_loader):
        self.epoch = epoch
        if mode in ['train']:
            self.model.train()
            self.lr_scheduler.step()
            dataloader = train_loader
        else:
            assert mode == 'val'
            self.model.eval()
            dataloader = val_loader
        self.__solve__(mode, epoch, dataloader)




