title: Word Embedings with word2vec
tags: deep-learning,word2vec,SoT
time: 19-7-1 8:00PM
slug: word2vec
ipython: ipynb/word2vec.ipynb
hide:

When working with MNIST, we've used one-hot vectors to specify digits when inputting them to our neural net. This makes a fair amount of sense for digits, since they're fairly distinct and there aren't many of them. When working with words, however, there are a lot of them<sup>[citation needed]</sup>. We could use a one-hot vector, but it'd have to be as large as our vocabulary is, which means a lot of parameters to tune or a small vocabulary. However, unlike the digits in MNIST, there is a lot to the relationships between words. The words "happy" and "joyful" are very similar while "apple" is very different. The idea behind word embeddings is creating vectors of a fixed size that represent different words where "similar" words have closer vectors.