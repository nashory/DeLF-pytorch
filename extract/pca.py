
'''pca.py
calculate PCA for given features, and save output into file.
'''

import os
import sys
import time
import glob

import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DelfPCA():
    def __init__(self,
                 pca_n_components,
                 pca_whitening=True,
                 pca_parameters_path='./output/pca/pca.h5'):
        self.pca_n_components = pca_n_components
        self.pca_whitening = pca_whitening
        self.pca_parameters_path = pca_parameters_path

    def __call__(self,
                 feature_maps):
        '''training pca.
        Args:
            feature_maps: list of feature tensorsm,
                          feature_maps = [f1, f2, f3 ...],
                          f1 = FloatTensor(fmap_depth)
        Returns:
            pca_matrix,
            pca_means,
            pca_vars
        '''

        # calculate pca.
        pca = PCA(whiten=self.pca_whitening)
        pca.fit(np.array(feature_maps))
        pca_matrix = pca.components_
        pca_mean = pca.mean_
        pca_vars = pca.explained_variance_
        
        # save as h5 file.
        print('================= PCA RESULT ==================')
        print('pca_matrix: {}'.format(pca_matrix.shape))
        print('pca_mean: {}'.format(pca_mean.shape))
        print('pca_vars: {}'.format(pca_vars.shape))
        print('===============================================')
        
        # save features, labels to h5 file.
        filename = os.path.join(self.pca_parameters_path)
        h5file = h5py.File(filename, 'w')
        h5file.create_dataset('pca_matrix', data=pca_matrix)
        h5file.create_dataset('pca_mean', data=pca_mean)
        h5file.create_dataset('pca_vars', data=pca_vars)
        h5file.close()

