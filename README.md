# Rethinking the Value of Network Pruning
This repository contains the code for reproducing the results, and trained ImageNet models, in the following arXiv paper:  

[Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270) 

[Zhuang Liu](https://liuzhuang13.github.io/)\*, Mingjie Sun\*, [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Gao Huang](http://www.gaohuang.net/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (\* equal contribution).

This paper shows that training the small pruned model from scratch can almost always achieve comparable or higher level of accuracy than the model obtained from the typical "training, pruning and fine-tuning" procedure. Our results suggest:  

1. Training a large, over-parameterized model is not necessary to obtain an efficient final model. 
2. Learned "important" weights of the large model are not necessarily useful for the small pruned model. 
3. The pruned architecture itself, rather than a set of inherited "important" weights, is what leads to the efficiency benefit in the final model, which suggests that some pruning algorithms could be seen as performing network architecture search.

## Citation

```
@article{liu2018rethinking,
  title={Rethinking the Value of Network Pruning},
  author={Zhuang Liu and Mingjie Sun and Tinghui Zhou and Gao Huang and Trevor Darrell},
  journal={arXiv preprint arXiv:1810.05270},
  year={2018}
}
```

## Implementation
We evaluated the following six pruning methods:  

1. [L1-norm based channel pruning](https://arxiv.org/abs/1608.08710)
2. [ThiNet](https://arxiv.org/abs/1707.06342)
3. [Regression based feature reconstruction](https://arxiv.org/abs/1707.06168)
4. [Network Slimming](https://arxiv.org/abs/1708.06519)
5. [Sparse Structure Selection](https://arxiv.org/abs/1707.01213)
6. [Non-structured weight-level pruning](https://arxiv.org/abs/1506.02626)

For CIFAR, our code is based on [pytorch-classification](https://github.com/bearpaw/pytorch-classification) and [network-slimming](https://github.com/Eric-mingjie/network-slimming). For ImageNet, we use the [official Pytorch ImageNet training code](https://github.com/pytorch/examples/blob/0.3.1/imagenet/main.py). The instructions and models are in each subfolder.

Our experiment environment is Python 3.6 & PyTorch 0.3.1.

## Contact
sunmj15 at gmail.com  
liuzhuangthu at gmail.com
