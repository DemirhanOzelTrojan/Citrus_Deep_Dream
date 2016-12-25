# Deep_Dream

##What is deep dreaming?
Deep dreaming is a computational concept created by Google which  uses a convolutional neural network to find and enhance patterns in images via algorithmic pareidolia, thus creating a dreamlike hallucinogenic appearance in the deliberately over-processed images.

##What is Deep_Dream?

Deep_Dream is a python script which was written to use the pretrained "inception" neural network by google in order to find image similarities and output "deep dream images"

##Examples:

### Original Image:
![Alt text](http://i.imgur.com/6FAzGVY.png "")

### Result: 
![Alt text](http://i.imgur.com/KPBuO4S.png "")


##Explanation:

      def render_deepdream(t_obj, img0=img_noise,
                           iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
          t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
          t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

          # split the image into a number of octaves
          img = img0
          octaves = []
          for _ in range(octave_n - 1):
              hw = img.shape[:2]
              lo = resize(img, np.int32(np.float32(hw) / octave_scale))
              hi = img - resize(lo, hw)
              img = lo
              octaves.append(hi)

          # generate details octave by octave
          for octave in range(octave_n):
              if octave > 0:
                  hi = octaves[-octave]
                  img = resize(img, hi.shape[:2]) + hi
              for _ in range(iter_n):
                  g = calc_grad_tiled(img, t_grad)
                  img += g * (step / (np.abs(g).mean() + 1e-7))


The render_deepdream() function takes an image parameter and a number of steps and iterations as well as a scale for resizing octaves. Alongside it's other parameters it takes a tensor from the inception model at which it renders the image and then stores it. 

render_deepdream () takes the input image splits it into octaves and resizes them so that the layers of inception model can search for similarities with smaller data for more accurate results. After an octave has been rendered, the data is stored in the layer in which it was rendered, and then the octace is re-rendered to find maximum similarity with the data that inception already knows.

All octaves are stored in an array and then matplotlib is used to arrange the octaves to form a proper image.

  
     render_deepdream(tf.square(T('mixed4c')), img0)
  

This function will render what the network sees at the specified "layer" for T. This can be modified by simply replacing the parameter. For a list of all valid inception layers, pleas visit: data/models.txt.

##User Guide:

 
    import matplotlib.pyplot as plt
    import numpy as np
    import PIL.Image
    import tensorflow as tf
    import os
    import zipfile


Please make sure that all coresponding packages have been installed through an appropriate python package manager.


To install tensorflow, please issue the following command in your command prompt.

#### For GPU Tensorflow: (reccomended)

pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-0.12.0-cp35-cp35m-win_amd64.whl


for GPU tensorflow, please ensure that you are using the proper Nvidia Nsight drivers, and the latest CUDNN package by Nvidia.

#### For CPU Tensorflow: 

pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.0-cp35-cp35m-win_amd64.whl





