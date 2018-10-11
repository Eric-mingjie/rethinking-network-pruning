# Non-Structured Pruning/Weight-Level Pruning

This directory contains a pytorch implementation of the CIFAR experiments of non-structured pruning introduced in this [paper](https://arxiv.org/abs/1506.02626) (NIPS 2015).

## Dependencies
progress v1.3, torch v0.3.1, torchvision v0.2.0

## Implementation
We prune only the weights in the convolutional layer. We use the mask implementation, where during pruning, we set the weights that are pruned to be 0. During training, we make sure that we don't update those pruned parameters.

## Baseline 

The `dataset` argument specifies which dataset to use: `cifar10` or `cifar100`. The `arch` argument specifies the architecture to use: `vgg` or `resnet`. The depth is chosen to be the same as the networks used in the paper.
```shell
python cifar.py --dataset cifar10 --arch vgg19_bn --depth 19
python cifar.py --dataset cifar10 --arch preresnet --depth 110
python cifar.py --dataset cifar10 --arch densenet --depth 40
python cifar.py --dataset cifar10 --arch densenet --depth 100 --compressionRate 2
```

## Prune

```shell
python cifar_prune.py --arch vgg19_bn --depth 19 --dataset cifar10 --percent 0.3 --resume [PATH TO THE MODEL] --save_dir [DIRECTORY TO STORE RESULT]
python cifar_prune.py --arch preresnet --depth 110 --dataset cifar10 --percent 0.3 --resume [PATH TO THE MODEL] --save_dir [DIRECTORY TO STORE RESULT]
python cifar_prune.py --arch densenet --depth 40 --dataset cifar10 --percent 0.3 --resume [PATH TO THE MODEL] --save_dir [DIRECTORY TO STORE RESULT]
python cifar_prune.py --arch densenet --depth 100 --compressionRate 2 --dataset cifar10 --percent 0.3 --resume [PATH TO THE MODEL] --save_dir [DIRECTORY TO STORE RESULT]
```


## Fine-tune
```shell
python cifar_finetune.py --arch vgg19_bn --depth 19 --dataset cifar10  --resume [PATH TO THE PRUNED MODEL]
python cifar_finetune.py --arch preresnet --depth 110 --dataset cifar10  --resume [PATH TO THE PRUNED MODEL]
python cifar_finetune.py --arch densenet --depth 40 --dataset cifar10  --resume [PATH TO THE PRUNED MODEL]
python cifar_finetune.py --arch densenet --depth 100 --compressionRate 2 --dataset cifar10  --resume [PATH TO THE PRUNED MODEL]
```

## Scratch-E
```
python cifar_E.py --arch vgg19_bn --depth 19 --dataset cifar10 --scratch [PATH TO THE PRUNED MODEL]
python cifar_E.py --arch preresnet --depth 110 --dataset cifar10 --scratch [PATH TO THE PRUNED MODEL]
python cifar_E.py --arch densenet --depth 40 --dataset cifar10 --scratch [PATH TO THE PRUNED MODEL]
python cifar_E.py --arch densenet --depth 100 --compressionRate 2 --dataset cifar10 --scratch [PATH TO THE PRUNED MODEL]
```

## Scratch-B
```
python cifar_B.py--arch vgg19_bn --depth 19 --dataset cifar10 --scratch [PATH TO THE PRUNED MODEL]
python cifar_B.py--arch preresnet --depth 110 --dataset cifar10 --scratch [PATH TO THE PRUNED MODEL]
python cifar_B.py--arch densenet --depth 40 --dataset cifar10 --scratch [PATH TO THE PRUNED MODEL]
python cifar_B.py--arch densenet --depth 100 --dataset cifar10 --scratch [PATH TO THE PRUNED MODEL]
```
 
