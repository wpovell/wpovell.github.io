title: New Blog
tags: blog
time: 17-6-21 1:00AM
slug: new-blog

It seems like every few years I get dissatisfied with the blogging platform I'm using and make a new one. This time around I decided to go for a static site much like the first github site I made back in the day:

<img style="display:block;margin: auto; width:70%" src="/imgs/blog_pt1.png"></img>


I wanted to roll my own mainly because with my last blog I never got into customizing it much due to my lack of experience with PHP/Wordpress templates. For this blog I considered using [Pelican](http://docs.getpelican.com/en/stable/) and other static site generators, but they seemed like more than I needed. Currently the blog is just based around a few [Jinja2](http://jinja.pocoo.org/docs/2.9/) templates and a single generator script that I used to insert markdown files. The main hassle was that I wanted to include iPython notebooks in my posts, however the guy who runs [Pythonic Perambulations](http://jakevdp.github.io/) made a nice script that converts `.ipynb`s into clean HTML that can be inserted without much hassel.