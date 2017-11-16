title: Making a Pong AI with Policy Gradients
tags: deep-learning,PG,pong,SoT
time: 17-6-30 4:00PM
slug: pg-pong
ipython: ipynb/PG-Pong.ipynb

This post will walk through training a pong AI using policy gradients, a type of reenforcement learning. This model continually  plays against a hard coded AI, using a neural net to decide on the move it makes. As it plays, it's actions are rewarded if they produce a score and discouraged if the opponent scores. Over time the idea is that the AI's move will become better and better as good play is encouraged.

<video style="width: 20%;display: block;margin: auto;" autoplay loop><source src="/imgs/ipynb/pong_movie.mp4" type="video/mp4"/></video>

