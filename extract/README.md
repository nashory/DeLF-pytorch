
You can train pca and get delf feature files by simply changing the hyperparameters in `extract.py`.

### Hyperparameters
~~~
+ MODE:
  'pca' or 'delf'.  
  'pca': extract feature to get pca matrix.   
  'delf': extract delf feature and save it to file.  
+ USE_PCA:  
  if you want to use pca dimesion reduction when delf feature extraction.
  this flag is only for MODE='delf'.
  
+ PCA_DIMS:
  final dimension after dimension reductin by pca.

+ PCA_PARAMETERS_PATH:
  when MODE=='pca': where to save pca.h5 file. (pca.hy5 file includes calculated pca matrix, pca vars, pca means)
  when MODE='delf': where to load pca matrix from to extract delf feature.
  
+ INPUT_PATH:
  path to input image to extract feature.

+ OUTPUT_PATH:
  path to output delf feature file.
  this option is only for MODE='delf'
  
+ LOAD_FROM:
  path to pytorch model you wish to use as a feature extractor.
~~~


### How to train pca?
modify hyperparameters in `extract.py`,  
and run `python extract.py`

~~~
(example)
MODE = 'pca'
GPU_ID = 0
IOU_THRES = 0.98
ATTN_THRES = 0.17
TOP_K = 1000
USE_PCA = False
PCA_DIMS = 40
SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
ARCH = 'resnet50'
EXPR = 'dummy'
TARGET_LAYER = 'layer3'
LOAD_FROM = 'xxx'
PCA_PARAMETERS_PATH = 'xxx'
INPUT_PATH = 'xxx'
OUTPUT_PATH = 'dummy'

python extract.py
~~~


### How to extract delf feature?
modify hyperparameters in `extract.py`,  
and run `python extract.py`

~~~
(example)
MODE = 'delf'
GPU_ID = 0
IOU_THRES = 0.98
ATTN_THRES = 0.17
TOP_K = 1000
USE_PCA = True
PCA_DIMS = 40
SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
ARCH = 'resnet50'
EXPR = 'dummy'
TARGET_LAYER = 'layer3'
LOAD_FROM = 'yyy'
PCA_PARAMETERS_PATH = 'yyy'
INPUT_PATH = 'yyy'
OUTPUT_PATH = './output.delf'

python extract.py
~~~

### [!!!] BE CAREFUL [!!!]
+ SIZE LIMIT:    
  If width * height > 1400*1400, the feature will be passed to prevent GPU memory overflow (24GB).    
  Make sure the input image size(w x h) is less than 1400 * 1400.  

