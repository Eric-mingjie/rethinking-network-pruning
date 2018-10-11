# ThiNet
This directory contains a pytorch implementation of the ImageNet experiments of [ThiNet](https://arxiv.org/abs/1707.06342). The authors have released their code and models in this [repository](https://github.com/Roll920/ThiNet).

## Finetune
We use the released model from their repository, where they use Caffe. Therefore, we test the models in Caffe and report the accuracy in the paper.

## Scratch-E
```
python main_E.py --arch vgg16 --model thinet-conv --lr 0.01 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_E.py --arch vgg16 --model thinet-gap --lr 0.01 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_E.py --arch vgg16 --model thinet-tiny --lr 0.01 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_E.py --arch resnet50 --model thinet-30 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_E.py --arch resnet50 --model thinet-50 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_E.py --arch resnet50 --model thinet-70 --save [PATH TO SAVE MODEL] [IMAGENET]
```
Here, `thinet-conv`, `thinet-gap` , `thinet-tiny` , `thinet-30`, `thinet-50`, `thinet-70` refer to the models in ThiNet.

## Scratch-B
<!-- For scratch-B training, first calculate the flops of the pruned model using function `count_model_param_flops` in `compute_flops.py`. Then compute the total epochs in scratch-B and corresponding learning rate schedule. -->
```
python main_B.py --arch vgg16 --model thinet-conv --lr 0.01 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_B.py --arch vgg16 --model thinet-gap --lr 0.01 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_B.py --arch vgg16 --model thinet-tiny --lr 0.01 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_B.py --arch resnet50 --model thinet-30 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_B.py --arch resnet50 --model thinet-50 --save [PATH TO SAVE MODEL] [IMAGENET]
python main_B.py --arch resnet50 --model thinet-70 --save [PATH TO SAVE MODEL] [IMAGENET]
```
For all networks other than `thinet-70`, the number of epochs for scratch-B training is 180; for `thinet-70`, the number of epochs for scratch-B training is 141.

## Models
We test the model using the scheme: resize the shorter edge to 256 and center crop to (224,224).
### VGG
Network|Training method|Top-1|Top-5|Download
:---:|:---:|:---:|:---:|:---:
VGG-Conv|scratch-E| 68.76| 88.71| [pytorch model (1003 MB)](https://drive.google.com/open?id=1Jr7n5q4BiYEHUVEv1FuzfAn0S26CNFsd)
VGG-Conv|scratch-B| 71.72| 90.34| [pytorch model (1003 MB)](https://drive.google.com/open?id=12DC2hpbQNVUSpS3ojcxjncW69jh1NpbN)
VGG-GAP|scratch-E| 66.85| 87.07| [pytorch model (64 MB)](https://drive.google.com/open?id=1FnPVJGjlL36tOJo1__7nr3Jykk-VhMLv)
VGG-GAP|scratch-B| 68.66| 88.13| [pytorch model (64 MB)](https://drive.google.com/open?id=1YqDnc6JbXQl83E50P1fmUDO7J7SuEdgK)
VGG-Tiny|scratch-E| 57.15| 79.92| [pytorch model (10  MB)](https://drive.google.com/open?id=1J-ydiASraEdKYEwDu-u5kG8FFgdyXpV_)
VGG-Tiny|scratch-B| 59.93| 82.07| [pytorch model (10 MB)](https://drive.google.com/open?id=1J1JRBLd-2AbDNk57621Wst02QlrC4jf4)

### ResNet
Network|Training method|Top-1|Top-5|Download
:---:|:---:|:---:|:---:|:---:
ThiNet-30|scratch-E| 70.91| 90.14| [pytorch model (66  MB)](https://drive.google.com/open?id=14cJ_oF4bAatcXiKEhBQHWny5kiC0hMBu)
ThiNet-30|scratch-B| 71.57| 90.49| [pytorch model (66 MB)](https://drive.google.com/open?id=1RkiTHKxFfmP6jYl2vo_gYC2NZzwfIe1M)
ThiNet-50|scratch-E| 73.31| 91.49| [pytorch model (95  MB)](https://drive.google.com/open?id=1E3c_7wvGXeUywXYVWIhzup5rp47TB9Tw)
ThiNet-50|scratch-B| 73.90| 91.98| [pytorch model (95 MB)](https://drive.google.com/open?id=1-0ip4ZDSxpbQx7D_5-VgOs-B8ww2jNC4)
ThiNet-70|scratch-E| 74.42| 92.07| [pytorch model (130 MB)](https://drive.google.com/open?id=1rTdotQKYBVHr03n1kYjYAVLDwxyYJok1)
ThiNet-70|scratch-B| 75.14| 92.34| [pytorch model (130 MB)](https://drive.google.com/open?id=1p2ER072IyFmDZRoAdQrRsedLElnmuct4)
