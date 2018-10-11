# Sparse Structure Selection
The authors have released the [code](https://github.com/TuSimple/sparse-structure-selection) for [Data Driven Sparse Structure Selection for Deep Neural Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zehao_Huang_Data-Driven_Sparse_Structure_ECCV_2018_paper.pdf).

In this repository, we describe how we use the released code for our experiments. For the accuracy of the pruned model, we use the results of the original paper. 

1. Modify the file `config/cfgs.py`, set `sss=False`.
2. Modify the file `symbol/resnet.py` to support ResNet-41, ResNet-32, ResNet-26 from the original paper. The details are as follows: I add a new parameter in function [residual_unit](https://github.com/TuSimple/sparse-structure-selection/blob/master/symbol/resnet.py#L10) in `symbol/resnet.py` to indicate whether this block's residual connection is pruned. In this way, we can modify the code [here](https://github.com/TuSimple/sparse-structure-selection/blob/master/symbol/resnet.py#L10) to support all pruned model of ResNet.  
It would be helpful to create a new `config.units` format as follows:  
    a. ResNet-41:  `[(0,False), (4,True), (6,True),(3,True)]`  
    b. ResNet-32:  `[(1,False), (4,True), (4,True),(1,True)]`  
    c. ResNet-26:  `[(0,False), (2,False), (5,False),(1,True)]`  
where (a,b) is for each stage and `a` means that this stage has `a` blocks remaining and `b` is True means that the first block in this stage is not pruned. (The reason why we make a distinction for the first block is that the first block in each stage contains a downsample convolution which is a corner case in the code.)  
3. Training: `python train.py`. Specify the gpu configuration in `config/cfgs.py`. For scratch-E training, use the standard 100 epochs with learning rate decay at 30, 60, 90 epochs. Also, for scratch-B training, modify the `lr_step` in `config/cfgs.py`, where each learning rate stage is expanded with a uniform ratio (FLOPs reduction ratio).
### Scratch-B training schedule
Network|Epochs|lr step|
:---:|:---:|:---:|
ResNet-41|117| [35, 70, 105]
ResNet-32|145| [43, 86, 129]
ResNet-26|179| [53, 106, 159]

## Models
Network|Training method|Top-1|Top-5|Download
:---:|:---:|:---:|:---:|:---:
ResNet-26|scratch-E| 72.31| 90.90| [MXNET model (57 MB)](https://drive.google.com/open?id=1wWPu5EGyT9lzKpYhqG2_1emdE59ctmvb)
ResNet-26|scratch-B| 73.41| 91.39| [MXNET model (57 MB)](https://drive.google.com/open?id=1lC5TXPtz_9Py3yeOA_MGXvXwBiL6wdYS)
ResNet-32|scratch-E| 73.79| 91.80| [MXNET model (55 MB)](https://drive.google.com/open?id=1h8iwPIT8z3h8ETFGP740FeEgYfxteLqU)
ResNet-32|scratch-B| 74.67| 92.22| [MXNET model (55 MB)](https://drive.google.com/open?id=1ud-K1p_g7ltD3MTJqiqkFgpLJeN_yAVB)
ResNet-41|scratch-E| 75.70| 92.74| [MXNET model (130 MB)](https://drive.google.com/open?id=1DgaqzjMqiFZz1vftKSw8yESCrrp6R6QV)
ResNet-41|scratch-B| 76.17| 92.90| [MXNET model (130 MB)](https://drive.google.com/open?id=1DgaqzjMqiFZz1vftKSw8yESCrrp6R6QV)
