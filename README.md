
# Pytorch Implementation of Deep Local Feature (DeLF)
reference: https://arxiv.org/pdf/1612.06321.pdf


## Prerequisites
+ PyTorch
+ python3
+ CUDA

## Training DeLF
There are 2 steps for DeLF training: (1) finetune stage, and (2) keypoint stage.  
Finetune stage loads resnet50 model pretrained on ImageNet, and finetune.  
Keypoint stage freezes the "base" network, and only update "attention" network for keypoint selection.
After the train process is done, model will be saved at `repo/<expr>/keypoint/ckpt`

### (1) training finetune stage:
~~~shell
$ cd train/
$ python main.py \
    --stage 'finetune' \
    --optim 'sgd' \
    --gpu_id 6 \
    --expr 'landmark' \
    --ncls 586 \
    --finetune_train_path <path to train data> \
    --finetune_val_path <path to val data> \
~~~

### (2) training keypoint stage:
+ load_from: absolute path to pytorch model you wish to load. (<model_name>.pth.tar)
+ expr: name of experiment you wish to save as.
~~~shell
$ cd train/
$ python main.py \
    --stage 'keypoint' \
    --gpu_id 6 \
    --ncls 586 \
    --optim 'sgd' \
    --use_random_gamma_scaling true \
    --expr 'landmark' \
    --load_from <path to model> \
    --keypoint_train_path <path to train data> \
    --keypoint_val_path <path to val data> \
~~~


## Feature Extraction of DeLF
There are also two steps to extract DeLF: (1) train PCA, (2) extract dimension reduced DeLF.  
__IMPORTANT: YOU MUST CHANGE OR COPY THE NAME OF MODEL from `repo/<expr>/keypoint/ckpt/bestshot.pth.tar` to `repo/<expr>/keypoint/ckpt/fix.pth.tar`.__  
__I intentionally added this to prevent the model from being updated after the PCA matrix is already calculated.__

### (1) train PCA
~~~shell
$ cd extract/
$ python extractor.py
    --gpu_id 4 \
    --load_expr 'delf' \
    --mode 'pca' \
    --stage 'inference' \
    --batch_size 1 \
    --input_path <path to train data>, but it is hardcoded.
    --output_path <output path to save pca matrix>, but it is hardcoded.
~~~

### (2) extract dimension reduced DeLF
~~~shell
$ cd extract/
$ python extractor.py
    --gpu_id 4 \
    --load_expr 'delf' \
    --mode 'delf' \
    --stage 'inference' \
    --batch_size 1 \
    --attn_thres 0.31 \
    --iou_thres 0.92 \
    --top_k 1000 \
    --use_pca True \
    --pca_dims 40 \
    --pca_parameters_path <path to pca matrix file.>, but it is hardcoded.
    --input_path <path to train data>, but it is hardcoded.
    --output_path <output path to save pca matrix>, but it is hardcoded.
~~~


## Visualization
You can visualize DeLF matching batween two arbitrary query images.
Let's assume there exist two images, test/img1.jpg, test/img2.jpg in `extract/test/` folder.
Run visualize.ipynb using Jupyter Notebook, and run each cells.
You may get the result like below.



## Author
Minchul Shin([@nashory](https://github.com/nashory))  
contact: min.nashory@navercorp.com   

![image](https://camo.githubusercontent.com/e053bc3e1b63635239e8a44574e819e62ab3e3f4/687474703a2f2f692e67697068792e636f6d2f49634a366e36564a4e6a524e532e676966)
