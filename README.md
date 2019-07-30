# pixel-cherry-picking
Autoencoding an image into latent variable and discrete mask images.

To run: 
```
$ python3 -u train_model1.py --gpu 0 --batchsize 128 --dim-z 32 --masks 4
```

Tested on: chainer(6.1.0) + cupy-cuda91(6.1.0) + matplotlib(3.0.3) + Python 3.6.5
