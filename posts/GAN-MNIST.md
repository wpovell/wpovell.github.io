title: Generating MNIST Digits with CGANs
tags: tensorflow,MNIST,GAN,CGAN,SoT
time: 17-6-25 4:00PM
slug: gan-mnist
ipython: ipynb/GAN-MNIST.ipynb

This post will walk through creating a Conditional Generative Adversarial Network (CGAN) to generate MNIST digits. The model works by having one neural net that generates an image, and another net that tries to determine if an image is real or generated. By competing, the generator slowly learns to make better and better "forgeries" of real MNIST digits.

<p style="text-align:center"><img src="/imgs/GAN-MNIST.gif"></img></p>