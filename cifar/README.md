# CIFAR Experiments
This directory contains all the CIFAR experiments in the paper, where there are four pruning methods in total:  

1. [L1-norm based channel pruning](https://arxiv.org/abs/1608.08710)
2. [Network Slimming](https://arxiv.org/abs/1708.06519)
3. [Soft filter pruning](https://www.ijcai.org/proceedings/2018/0309.pdf)
4. [Non-structured weight-level pruning](https://arxiv.org/abs/1506.02626)

For each method, we give example commands for baseline training, finetuning, scratch-E training and scratch-B training.  

We also give our implementation for [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635).

## Implementation
Our code is based on [network-slimming](https://github.com/Eric-mingjie/network-slimming) and [pytorch-classification](https://github.com/bearpaw/pytorch-classification).  
