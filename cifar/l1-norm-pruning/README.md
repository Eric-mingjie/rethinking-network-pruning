# Pruning Filters For Efficient ConvNets

This directory contains a pytorch re-implementation of all CIFAR experiments of the following paper  
[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) (ICLR 2017).

## Dependencies
torch v0.3.1, torchvision v0.2.0

## Baseline 

The `dataset` argument specifies which dataset to use: `cifar10` or `cifar100`. The `arch` argument specifies the architecture to use: `vgg` or `resnet`. The depth is chosen to be the same as the networks used in the paper.
```shell
python main.py --dataset cifar10 --arch vgg --depth 16
python main.py --dataset cifar10 --arch resnet --depth 56
python main.py --dataset cifar10 --arch resnet --depth 110
```

## Prune

```shell
python vggprune.py --dataset cifar10 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python res56prune.py --dataset cifar10 -v A --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python res110prune.py --dataset cifar10 -v A --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
Here in `res56prune.py` and `res110prune.py`, the `-v` argument is `A` or `B`, which refers to the naming of the pruned model in the original paper. The pruned model will be named `pruned.pth.tar`.

## Fine-tune

```shell
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 16 
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 56 
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 110 
```

## Scratch-E
```
python main_E.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 16
python main_E.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 56 
python main_E.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 110  
```

## Scratch-B
```
python main_B.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 16
python main_B.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 56
python main_B.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 110
```

