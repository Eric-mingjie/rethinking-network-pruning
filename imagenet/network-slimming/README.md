# Network Slimming 
This directory contains the code for implementing Network Slimming on ImageNet. 

## Implementation
We use the `mask implementation` for finetuning, where during pruning we set 0 to the channel scaling factor
whose corresponding channels are pruned. When finetuning the pruned model, in each iteration, before we call `optimizer.step()`, we update the gradient of those 0 scaling factors to be 0. This is achieved in `BN_grad_zero` function.

## Train with sparsity
```
python main.py --arch vgg11_bn --s 0.00001 --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Prune
```
python prune.py --arch vgg11_bn --percent 0.5 --model [PATH TO THE BASE MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Finetune
```
python main_finetune.py --arch vgg11_bn --refine [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Scratch-E
```
python main_E.py --arch vgg11_bn --scratch [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Scratch-B
```
python main_B.py --arch vgg11_bn --scratch [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Models
Network|Prune ratio|Training method|Top-1|Top-5|Download
:---:|:---:|:---:|:---:|:---:|:---:
VGG-11(mask-impl)|50%|finetune| 68.62| 88.77| [pytorch model (1014 MB)](https://drive.google.com/open?id=10uscgVM_5ghsxI110y5-sl8T3Kkzki6N)
VGG-11(mask-impl)|50%|scratch-E| 70.00| 89.33| [pytorch model (1014 MB)](https://drive.google.com/open?id=11ITIlGYUu9wZAF-sp06L5h5JoTKYtWsS)
VGG-11|50%|scratch-B| 71.18| 90.08| [pytorch model (282 MB)](https://drive.google.com/open?id=1HjCAETR2kAx2uORe9yxKXZxidxQJboQx)