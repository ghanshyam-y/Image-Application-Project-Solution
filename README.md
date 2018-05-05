# Image Classifier Application Project


## Training 
Use the following to train a densenet121 model

```python train.py flowers --arch densenet121 --gpu --epochs 4```

The arch argument "densenet121" can also be replaced with "vgg16" for a VGG16 model.

Try using --help for more details.

## Prediction

The checkpoint saved during the training phase can be used here.
To get the top 5 most likely classes try using

```python predict.py flowers/test/28/image_05230.jpg checkpoints/densenet121_epoch1.pth --gpu --top_k 5```

Try --help for more details.
