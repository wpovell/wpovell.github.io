title: Image to Image Translation with pix2pix
tags: deep-learning,image-translation,GAN,pix2pix,SoT
time: 19-7-21 1:00AM
slug: pix2pix
ipython: ipynb/pix2pix.ipynb

One cool application of [GANs](/posts/gan-mnist.html) is image translation. The task is given pairs of images, can you learn to generate the output image from an input. Examples of this task would be turning satelite images into maps or coloring black and white photos. In this post we're going to walk through the [pix2pix model by Isola et al.](https://arxiv.org/abs/1611.07004)

<figure>
<img style="width: 70%; display: block; margin:auto;" src="/imgs/ipynb/pix2pix/trans.jpg">
<figcaption style='text-align:center'>Image from the [pix2pix paper](https://arxiv.org/abs/1611.07004)</figcaption>
</figure>


