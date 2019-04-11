# Non-Structured Pruning/Weight-Level Pruning

This directory contains a pytorch implementation of the ImageNet experiments of non-structured pruning.

## Implementation
We prune only the weights in the convolutional layer. We use the mask implementation, where during pruning, we set the weights that are pruned to be 0. During training, we make sure that we don't update those pruned parameters.

## Baseline
We get the base model of VGG-16 and ResNet-50 from Pytorch [Model Zoo](https://pytorch.org/docs/stable/torchvision/models.html).

## Prune
```
python prune.py --arch vgg16_bn --pretrained --percent 0.3 --save [PATH TO SAVE RESULTS] [IMAGENET]
python prune.py --arch resnet50 --pretrained --percent 0.3 --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Finetune
```
python main_finetune.py --arch vgg16_bn --resume [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
python main_finetune.py --arch resnet50 --resume [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Scratch-E
```
python main_E.py --arch vgg16_bn --resume [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
python main_E.py --arch resnet50 --resume [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Scratch-B
```
python main_B.py --arch vgg16_bn --resume [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
python main_B.py --arch resnet50 --resume [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Models
### VGG
Network|Prune ratio|Training method|Top-1|Top-5|Download
:---:|:---:|:---:|:---:|:---:|:---:
VGG-16|30%|finetune| 73.68| 91.53| [pytorch model (1024 MB)](https://drive.google.com/open?id=1OWGaJ-tXAlS4Ne5zhZ1M4k-1rk723do0)
VGG-16|30%|scratch-E| 72.75| 91.06| [pytorch model (1024 MB)](https://drive.google.com/open?id=1kgGiBaG1Y6Kh-EK27APWoMzeV7jO_jlL)
VGG-16|30%|scratch-B| 74.02| 91.78| [pytorch model (1024 MB)](https://drive.google.com/open?id=1ADbEpkziEMs_FPKAP-6BytcBHfqshrlg)
VGG-16|60%|finetune| 73.63| 91.54| [pytorch model (1024 MB)](https://drive.google.com/open?id=1xZOFuxKJEdv9AtoHcv5VvZsrWM7-vujY)
VGG-16|60%|scratch-E| 71.50| 90.43| [pytorch model (1024 MB)](https://drive.google.com/open?id=1s4yETDG0WB7ZerHmGudVRo2Z0JWuxZXr)
VGG-16|60%|scratch-B| 73.42| 91.48| [pytorch model (1024 MB)](https://drive.google.com/open?id=1APsXiwxq2VCitKvGoeqfHieEdWEpMk6W)

### ResNet
Network|Prune ratio|Training method|Top-1|Top-5|Download
:---:|:---:|:---:|:---:|:---:|:---:
ResNet-50|30%|finetune| 76.06| 92.88| [pytorch model (195 MB)](https://drive.google.com/open?id=17bzfWtHjTkCture96d7MG0afrFiY9xFF)
ResNet-50|30%|scratch-E| 74.77| 92.19| [pytorch model (195 MB)](https://drive.google.com/open?id=1C3VxBlWbOwjtvlFe_5cRZpFFY0H4NJRp)
ResNet-50|30%|scratch-B| 75.60| 92.75| [pytorch model (195 MB)](https://drive.google.com/open?id=1z3ABz6Pk0drVueWJRucG68MeGnAyA6t7)
ResNet-50|60%|finetune| 76.09| 92.91| [pytorch model (195 MB)](https://drive.google.com/open?id=1iTwXpW61OodacsefyuSDljtGPj0FxvUY)
ResNet-50|60%|scratch-E| 73.69| 91.61| [pytorch model (195 MB)](https://drive.google.com/open?id=1LYyCHVypbkkS23RVOgcE8clUs3haRlHA)
ResNet-50|60%|scratch-B| 74.90| 92.28| [pytorch model (195 MB)](https://drive.google.com/open?id=17pqC05Sakt18xoRnpddAf-sYNs_vaKPk)