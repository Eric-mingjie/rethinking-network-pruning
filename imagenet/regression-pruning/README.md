# Channel Pruning for Accelerating Very Deep Neural Networks
This directory contains a pytorch implementation of the ImageNet experiments of this [paper](https://arxiv.org/abs/1707.06168). The authors have released their code and models in this [repository](https://github.com/yihui-he/channel-pruning).

## Implementation
For ResNet-2x, we introduce a `channel selection` layer for pruning the first convolutional layer in a residual block. The `indexes` of the `channel selection` layer is stored in `models/filter.pkl`, which is computed from the [official released model](https://github.com/yihui-he/channel-pruning/releases/tag/ResNet-50-2X). In loading the model ResNet-2x, the indexes in `filter.pkl` is automatically loaded into the network.

## Finetune
We use the released model from their repository, where they use Caffe. Therefore, we test the models in Caffe and report the accuracy in the paper.

## Scratch-E
```
python main_E.py --arch vgg16 --model vgg-5x --lr 0.01 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_E.py --arch resnet50 --model resnet-2x --save [PATH TO SAVE MODEL] [IMAGENET]
```

## Scratch-B
```
python main_B.py --arch vgg16 --model vgg-5x --lr 0.01 --save [PATH TO SAVE MODEL] [IMAGNET]
python main_B.py --arch resnet50 --model resnet-2x --save [PATH TO SAVE MODEL] [IMAGENET]
```
Here for VGG-2x, the number of epochs for scratch-B training is 180 epochs; for ResNet-2x, the number of epochs for scratch-B training (132 epochs) is computed according to the actual FLOPs reduction ratio.

## Models
We test the model using the scheme: resize the shorter edge to 256 and center crop to (224,224).

Network|Training method|Top-1|Top-5|Download
:---:|:---:|:---:|:---:|:---:
VGG-5x|scratch-E| 68.05| 88.15| [pytorch model (999 MB)](https://drive.google.com/open?id=151ysF8v39GuZHxAK9YjvTWoUqBqiIdJ0)
VGG-5x|scratch-B| 71.00| 89.96| [pytorch model (999 MB)](https://drive.google.com/open?id=1FiTQhRs4L19bp_YKoGXn2M_6BkxWSC_-)
ResNet-2x|scratch-E| 71.26| 90.68| [pytorch model (151 MB)](https://drive.google.com/open?id=1hdbcrB3-3z5n1WnQRJ6VfYeWrlYmDmkL)
ResNet-2x|scratch-B| 74.58| 92.23| [pytorch model (151 MB)](https://drive.google.com/open?id=16rgMbYHMtwl5rXOJHgKqyrTZB7O2sV40)