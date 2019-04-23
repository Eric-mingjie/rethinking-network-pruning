# Rethinking the Value of Network Pruning
This repository contains the code for reproducing the results, and trained ImageNet models, in the following paper:  

[Rethinking the Value of Network Pruning](https://openreview.net/pdf?id=rJlnB3C5Ym) 

[Zhuang Liu](https://liuzhuang13.github.io/)\*, Mingjie Sun\*, [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Gao Huang](http://www.gaohuang.net/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (\* equal contribution).

ICLR 2019. Also [Best Paper Award](https://nips.cc/Conferences/2018/Schedule?showEvent=10941) at NIPS 2018 Workshop on Compact Deep Neural Networks.

Our paper shows that for structured pruning, training the small pruned model from scratch can almost always achieve comparable or higher level of accuracy than the model obtained from the typical "training, pruning and fine-tuning" procedure. For those pruning methods:

1. Training a large, over-parameterized model is not absolutely necessary to obtain an efficient final model.
2. Learned “important” weights of the large model are typically not useful for the small pruned model. 
3. The pruned architecture itself, rather than a set of inherited “important” weights, is more crucial to the efficiency in the final model, which suggests that in some cases pruning can be useful as an architecture search paradigm. 

Our results suggest the need for more careful baseline evaluations in future research on structured pruning methods. 

## Citation

```
@inproceedings{liu2018rethinking,
  title={Rethinking the Value of Network Pruning},
  author={Zhuang Liu and Mingjie Sun and Tinghui Zhou and Gao Huang and Trevor Darrell},
  booktitle={ICLR},
  year={2019}
}
```

## Implementation
We evaluated the following seven pruning methods:  

1. [L1-norm based channel pruning](https://arxiv.org/abs/1608.08710)
2. [ThiNet](https://arxiv.org/abs/1707.06342)
3. [Regression based feature reconstruction](https://arxiv.org/abs/1707.06168)
4. [Network Slimming](https://arxiv.org/abs/1708.06519)
5. [Sparse Structure Selection](https://arxiv.org/abs/1707.01213)
6. [Non-structured weight-level pruning](https://arxiv.org/abs/1506.02626)
7. [Soft filter pruning](https://www.ijcai.org/proceedings/2018/0309.pdf)

For CIFAR, our code is based on [pytorch-classification](https://github.com/bearpaw/pytorch-classification) and [network-slimming](https://github.com/Eric-mingjie/network-slimming). For ImageNet, we use the [official Pytorch ImageNet training code](https://github.com/pytorch/examples/blob/0.3.1/imagenet/main.py). The instructions and models are in each subfolder.

Our experiment environment is Python 3.6 & PyTorch 0.3.1.

## Contact
sunmj15 at gmail.com  
liuzhuangthu at gmail.com
