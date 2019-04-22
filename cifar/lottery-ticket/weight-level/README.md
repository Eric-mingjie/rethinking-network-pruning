# The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

This directory contains a pytorch implementation of [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) for non-structured weight pruning introduced in this [paper](https://arxiv.org/abs/1506.02626) (NIPS 2015).

## Dependencies
torch v0.3.1, torchvision v0.2.0

## Baseline 

```shell
python cifar.py --dataset cifar10 --arch vgg16_bn --depth 16 \
    --lr 0.1 --save_dir [PATH TO SAVE THE MODEL]
```
Note that the initialization is stored in a file called `init.pth.tar`, which will be used when training the lottery ticket.

## Iterative Prune

```shell
python cifar_prune_iterative.py --dataset cifar10 --arch vgg16_bn --depth 16 \
    --percent RATIO --resume [PATH TO THE MODEL TO BE PRUNED] \
    --save [DIRECTORY TO STORE RESULT]
```
Note that `cifar_prune_iterative` is implemented as pruning all the nonzero element in the model and the ratio in `--percent` refers to the prune ratio respect to the total number of nonzero element. When a model is iteratively pruned, you just need to pass the model to be pruned each iteration to `--resume` and set the ratio to be the prune ratio respectively.

## Lottery Ticket

```shell
python lottery_ticket.py --dataset cifar10 --arch vgg16_bn --depth 16 \
    --lr 0.1 --resume [PATH TO THE PRUNED MODEL] \
    --model [PATH TO THE STORED INITIALIZATION] \
    --save_dir [PATH TO SAVE THE MODEL]
```

## Scratch-E
```
python cifar_scratch_no_longer.py --dataset cifar10 --arch vgg16_bn --depth 16 \
    --lr 0.1 --resume [PATH TO THE PRUNED MODEL] \
    --save_dir [PATH TO SAVE THE MODEL]
```

