# Pruning Filters for Efficient Convnets

## Baseline
We get the ResNet-34 baseline model from Pytorch model zoo.

## Prune
```
python prune.py -v A --save [PATH TO SAVE RESULTS] [IMAGENET]
python prune.py -v B --save [PATH TO SAVE RESULTS] [IMAGENET]
```
Here `-v` specifies the pruned model: ResNet-34-A or ResNet-34-B.

## Finetune
```
python main_finetune.py --arch resnet34 --refine [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Scratch-E
```
python main_E.py --arch resnet34 --scratch [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Scratch-B
```
python main_B.py --arch resnet34 --scratch [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

## Models
Network|Training method|Top-1|Top-5|Download
:---:|:---:|:---:|:---:|:---:
ResNet34-A|finetune| 72.56| 90.99| [pytorch model (154 MB)](https://drive.google.com/open?id=1EmxoTa0kCHSsFBpL8jEZ5CVF6_4bkxGM)
ResNet34-A|scratch-E| 72.77| 91.20| [pytorch model (154 MB)](https://drive.google.com/open?id=1f-x3XHBFpCbUM5Y3cuH1_J9lN4CcbZ_0)
ResNet34-A|scratch-B| 73.08| 91.29| [pytorch model (154 MB)](https://drive.google.com/open?id=1fQT68PATrGk9zt6HXgCzTNr-zsS9aLeq)
ResNet34-B|finetune| 72.29| 90.72| [pytorch model (149 MB)](https://drive.google.com/open?id=1pW05JPHAPd_-bR862CmQPsP_q1_jsRwG)
ResNet34-B|scratch-E| 72.55| 91.07| [pytorch model (149 MB)](https://drive.google.com/open?id=1YPcKrh1ctxUYsn2Yk-D4cuW7DVV1hX4-)
ResNet34-B|scratch-B| 72.84| 91.19| [pytorch model (149 MB)](https://drive.google.com/open?id=1f_Nl-bcxBdhp3R2bY1nby4PIbwHOTaUv)
