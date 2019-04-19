# Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks

This directory contains the pytorch implementation for [soft filter pruning](https://www.ijcai.org/proceedings/2018/0309.pdf) (IJCAI 2018).  

## Dependencies
- torch v0.3.1, torchvision v0.3.0

## Overview
Specify the path to dataset in `DATA`. The argument `--arch` can be [`resnet20`,`resnet32`,`resnet56`,`resnet110`].  
Below shows the choice of the argument `--layer_end` over different architectures:  
`resnet20`: 54  `resnet32`: 90 `resnet56`: 162 `resnet110`:324  
The hyperparameter settings are the same as that in the original paper.

## Baseline 
```shell
python pruning_cifar10_pretrain.py DATA --dataset cifar10 \
    --arch resnet56 --save_path [PATH TO SAVE THE MODEL] \
    --epochs 200 --schedule 1 60 120 160 --gammas 10 0.2 0.2 0.2 \
    --learning_rate 0.01 --decay 0.0005 --batch_size 128 --rate 0.7 \
    --layer_begin 0  --layer_end 162 --layer_inter 3 --epoch_prune 1 
```

## Soft Filter Pruning
By not passing the argument `--resume`, we do not use the pretrained model. To use pretrained models, pass the argument `--resume` with the path to the pretrained model.
```shell
python pruning_cifar10_resnet.py DATA --dataset cifar10 \
    --arch resnet56 --save_path [PATH TO SAVE THE PRUNED MODEL] \
    --epochs 200 --schedule 1 60 120 160 --gammas 10 0.2 0.2 0.2 \
    --learning_rate 0.001 --decay 0.0005 --batch_size 128 --rate 0.7 \
    --layer_begin 0  --layer_end 162 --layer_inter 3 --epoch_prune 1 
```

## Scratch-E
```shell
python pruning_resnet_scratch.py DATA --dataset cifar10 \
    --arch resnet56 --resume [PATH TO THE PRUNED MODEL] \
    --save [PATH TO SAVE THE MODEL] \
    --epochs 200 --schedule 1 60 120 160 --gammas 10 0.2 0.2 0.2 \
    --learning_rate 0.01 --decay 0.0005 --batch_size 128 --rate 0.7 
    --layer_begin 0 --layer_end 162 --layer_inter 3 --epoch_prune 1 
```

## Scratch-B
```shell
python pruning_resnet_longer_scratch.py DATA --dataset cifar10 \
    --arch resnet56 --resume [PATH TO THE PRUNED MODEL] \
    --save [PATH TO SAVE THE MODEL] \
    --epochs 200 --schedule 1 60 120 160 --gammas 10 0.2 0.2 0.2 \
    --learning_rate 0.01 --decay 0.0005 --batch_size 128 --rate 0.7 
    --layer_begin 0 --layer_end 162 --layer_inter 3 --epoch_prune 1 
```