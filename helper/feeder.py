'''feeder.py
'''

import os, sys, time
sys.path.append('../')
import argparse

from PIL import Image
import h5py
import torch
import torchvision.transforms as transforms

import helper.delf_helper as delf_helper
from train.delf import Delf_V1

__DEBUG__ = False

def __cuda__(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

def __build_delf_config__(data):
    parser = argparse.ArgumentParser('delf-config')
    parser.add_argument('--stage', type=str, default='inference')
    parser.add_argument('--expr', type=str, default='dummy')
    parser.add_argument('--ncls', type=str, default='dummy')
    parser.add_argument('--use_random_gamma_rescale', type=str, default=False)
    parser.add_argument('--arch', type=str, default=data['ARCH'])
    parser.add_argument('--load_from', type=str, default=data['LOAD_FROM'])
    parser.add_argument('--target_layer', type=str, default=data['TARGET_LAYER'])
    delf_config, _ = parser.parse_known_args()
    
    # print config.
    state = {k: v for k, v in delf_config._get_kwargs()}
    print(state)
    return delf_config


class Feeder():
    def __init__(self,
                 feeder_config):
        # environment setting.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(feeder_config.get('GPU_ID'))
        
        # parameters.
        self.iou_thres = feeder_config.get('IOU_THRES')
        self.attn_thres = feeder_config.get('ATTN_THRES')
        self.top_k = feeder_config.get('TOP_K')
        self.target_layer = feeder_config.get('TARGET_LAYER')
        self.scale_list = feeder_config.get('SCALE_LIST')
        self.workers = feeder_config.get('WORKERS')

        # load pytorch model
        print('load DeLF pytorch model...')
        delf_config = __build_delf_config__(feeder_config) 
        self.model = Delf_V1(
            ncls = delf_config.ncls,
            load_from = delf_config.load_from,
            arch = delf_config.arch,
            stage = delf_config.stage,
            target_layer = delf_config.target_layer,
            use_random_gamma_rescale = False)
        self.model.eval()
        self.model = __cuda__(self.model)
        
        # load pca matrix
        print('load PCA parameters...')
        h5file = h5py.File(feeder_config.get('PCA_PARAMETERS_PATH'), 'r')
        self.pca_mean = h5file['.']['pca_mean'].value
        self.pca_vars = h5file['.']['pca_vars'].value
        self.pca_matrix = h5file['.']['pca_matrix'].value
        self.pca_dims = feeder_config.get('PCA_DIMS')
        self.use_pca = feeder_config.get('USE_PCA')

        # !!! stride value in tensorflow inference code is not applicable for pytorch, because pytorch works differently.
        # !!! make sure to use stride=16 for target_layer=='layer3'.
        if self.target_layer in ['layer3']:
            self.fmap_depth = 1024
            self.rf = 291.0
            self.stride = 16.0
            self.padding = 145.0
        elif self.target_layer in ['layer4']:
            self.fmap_depth = 2048
            self.rf = 483.0
            self.stride = 32.0
            self.padding = 241.0
        else:
            raise ValueError('Unsupported target_layer: {}'.format(self.target_layer))
        

    def __resize_image__(self, image, target_size):
        return 'resize image.'

    def __transform__(self, image):
        transform = transforms.ToTensor()
        return transform(image)

    def __print_result__(self, data):
        print('----------------------------------------------------------')
        print('filename: ', data['filename'])
        print("location_np_list shape: ", data['location_np_list'].shape)
        print("descriptor_np_list shape: ", data['descriptor_np_list'].shape)
        print("feature_scale_np_list shape: ", data['feature_scale_np_list'].shape)
        print("attention_score_np_list shape: ", data['attention_score_np_list'].shape)
        print("attention_np_list shape: ", data['attention_np_list'].shape)
        print('----------------------------------------------------------')

    def __get_result__(self,
                       path,
                       image):
        # load tensor image
        x = __cuda__(self.__transform__(image))
        x = x.unsqueeze(0)

        # extract feature.
        data = delf_helper.GetDelfFeatureFromMultiScale(
            x = x,
            model = self.model,
            filename = path,
            pca_mean = self.pca_mean,
            pca_vars = self.pca_vars,
            pca_matrix = self.pca_matrix,
            pca_dims = self.pca_dims,
            rf = self.rf,
            stride = self.stride,
            padding = self.padding,
            top_k = self.top_k,
            scale_list = self.scale_list,
            iou_thres = self.iou_thres,
            attn_thres = self.attn_thres,
            use_pca = self.use_pca,
            workers = self.workers)
        
        if __DEBUG__:
            self.__print_result__(data)
        return data 

    def feed(self, pil_image, filename='dummy'):
        return self.__get_result__(filename, pil_image)

    def feed_to_compare(self, query_path, pil_image):
        '''feed_to_compare
        used to visualize mathcing between two query images.
        '''
        assert len(pil_image)==2, 'length of query list should be 2.'
        outputs = []
        for i in range(2):
            outputs.append(self.__get_result__(query_path[i], pil_image[i]))
        return outputs

