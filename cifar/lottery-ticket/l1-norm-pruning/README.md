# The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

This directory contains a pytorch implementation of [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) for non-structured weight pruning introduced in this [paper](https://arxiv.org/abs/1506.02626) (NIPS 2015).

## Dependencies
torch v0.3.1, torchvision v0.2.0

## Overview
Since lottery ticket hypothesis involves the initialization of baseline model before training, it is more easy to implement it using mask implementation (More explanation [here](https://github.com/Eric-mingjie/network-slimming/tree/master/mask-impl#mask-implementation-of-network-slimming)). 

## Baseline 

```shell
python main.py --arch vgg --depth 16 --dataset cifar10 \
    --lr 0.1 --save [PATH TO SAVE THE MODEL]
```
Note that the initialization is stored in a file called `init.pth.tar`, which will be used when training the lottery ticket.

## Iterative Prune

```shell
python lottery_vggprune.py --dataset cifar10 --model [PATH TO THE MODEL] --save [DIRECTORY TO SAVE THE MODEL]
python lottery_res56prune.py --dataset cifar10 -v A --model [PATH TO THE MODEL] --save [DIRECTORY TO SAVE THE MODEL]
python lottery_res110prune.py --dataset cifar10 -v A --model [PATH TO THE MODEL] --save [DIRECTORY TO SAVE THE MODEL]
```

## Lottery Ticket

```shell
python main_lottery.py --dataset cifar10 --arch vgg --depth 16 \
    --lr 0.1 --resume [PATH TO THE PRUNED MODEL] \
    --model [PATH TO THE STORED INTIALIZATION] \
    --save [PATH TO SAVE THE MODEL]
```

## Scratch-E
```
python main_scratch_mask.py --dataset cifar10 --arch vgg --depth 16 \
    --lr 0.1 --resume [PATH TO THE PRUNED MODEL] \
    --save [PATH TO SAVE THE MODEL]
```

