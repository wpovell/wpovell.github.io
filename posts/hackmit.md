title: HackMIT CAPTCHA Detection
tags: tensorflow,CNN,HackMIT
time: 17-7-6 1:30PM
slug: captcha-hackmit
ipython: ipynb/captcha_hackmit.ipynb
hide:

[HackMIT](http://hackmit.org) opened its registration recently and as a tradition offered a series of admission puzzles. There were five total involving various CS topics, including crypto and machine learning. The last puzzle was definitely my favorite so I thought I'd walk through my solution. The challenge was as follows: Given a CAPTCHA like

<img style='display:block;margin: auto;width: 30%' src="/imgs/captcha.png">

find the solution text. You could request as many images as you'd like, you just had to return 15k solutions with at least 10k correct (~66% accuracy). It was suggested that there was a deterministic way to get the CAPTCHAs from the IDs, however I went with the more straightword ML route.


