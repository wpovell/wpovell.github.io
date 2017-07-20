title: Style Transfer
tags: deep-learning,CNN,style-transfer,SoT
time: 17-6-24 4:00PM
slug: style-transfer
ipython: ipynb/style.ipynb

In this post I'm going to walk through Style Transfer using the method described in [Gatys et al. 2015](https://arxiv.org/pdf/1508.06576v2.pdf) with the [VGG Very-Deep 19](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) model as the source for filter weights.

We will be taking two images and attemp to transfer the style of one to the other, without losing the content of the recieving image. See the following example (more included at bottom of the post):

| Content                            | Style                             | Output                    |
|:----------------------------------:|:---------------------------------:|:-------------------------:|
| <img style='width: 100%' src="/imgs/ipynb/blueno.jpg"> | <img style='width: 100%' src="/imgs/ipynb/starry_night.jpg"> | <img style='width: 100%' src="/imgs/ipynb/out1.png"> |