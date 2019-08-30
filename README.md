# Rethinking the Value of Network Pruning
This repository contains the code for reproducing the results, and trained ImageNet models, in the following paper:  

Rethinking the Value of Network Pruning. [[arXiv]](https://arxiv.org/abs/1810.05270) [[OpenReview]](https://openreview.net/forum?id=rJlnB3C5Ym)

[Zhuang Liu](https://liuzhuang13.github.io/)\*, Mingjie Sun\*, [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Gao Huang](http://www.gaohuang.net/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (\* equal contribution).

ICLR 2019. Also [Best Paper Award](https://nips.cc/Conferences/2018/Schedule?showEvent=10941) at NIPS 2018 Workshop on Compact Deep Neural Networks.

Several pruning methods' implementations contained in this repo can also be readily used for other research purposes.

## Paper Summary

Our paper shows that for structured pruning, training the small pruned model from scratch can almost always achieve comparable or higher level of accuracy than the model obtained from the typical "training, pruning and fine-tuning" procedure. For those pruning methods:

1. Training a large, over-parameterized model is not absolutely necessary to obtain an efficient final model.
2. Learned “important” weights of the large model are typically not useful for the small pruned model. 
3. The pruned architecture itself, rather than a set of inherited “important” weights, is more crucial to the efficiency in the final model, which suggests that in some cases pruning can be useful as an architecture search paradigm. 

Our results suggest the need for more careful baseline evaluations in future research on structured pruning methods. 

We also compare with the "[Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)" (Frankle & Carbin 2019), and find that with optimal learning rate, the "winning ticket" initialization as used in Frankle & Carbin (2019) does not bring improvement over random initialization. For more details please refer to our paper.

## Implementation
We evaluated the following seven pruning methods. 

1. [L1-norm based channel pruning](https://arxiv.org/abs/1608.08710)
2. [ThiNet](https://arxiv.org/abs/1707.06342)
3. [Regression based feature reconstruction](https://arxiv.org/abs/1707.06168)
4. [Network Slimming](https://arxiv.org/abs/1708.06519)
5. [Sparse Structure Selection](https://arxiv.org/abs/1707.01213)
6. [Soft filter pruning](https://www.ijcai.org/proceedings/2018/0309.pdf)
7. [Unstructured weight-level pruning](https://arxiv.org/abs/1506.02626)

The first six is structured while the last one is unstructured (or sparse). For CIFAR, our code is based on [pytorch-classification](https://github.com/bearpaw/pytorch-classification) and [network-slimming](https://github.com/Eric-mingjie/network-slimming). For ImageNet, we use the [official Pytorch ImageNet training code](https://github.com/pytorch/examples/blob/0.3.1/imagenet/main.py). The instructions and models are in each subfolder.

For experiments on [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635), please refer to the folder [cifar/lottery-ticket](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/cifar/lottery-ticket).

Our experiment environment is Python 3.6 & PyTorch 0.3.1. 

## Contact
Feel free to discuss papers/code with us through issues/emails!

sunmj15 at gmail.com  
liuzhuangthu at gmail.com

## Citation
If you use our code in your research, please cite:
```
@inproceedings{liu2018rethinking,
  title={Rethinking the Value of Network Pruning},
  author={Zhuang Liu and Mingjie Sun and Tinghui Zhou and Gao Huang and Trevor Darrell},
  booktitle={ICLR},
  year={2019}
}
```
