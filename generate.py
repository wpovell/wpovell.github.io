from datetime import datetime
import pypandoc
import os
import jinja2
from collections import Counter

import ipynb

class Post:
	def __init__(self, fn):
		with open(fn, encoding='utf-8') as f:
			lines = list(f)

		addToHeader = True
		self.body = ''
		self.meta = {}
		for line in lines:
			if addToHeader and line == '\n':
				addToHeader = False
			elif addToHeader:
				tag, value = line.split(':', 1)
				tag = tag.strip().lower()
				value = value.strip()
				self.meta[tag] = value
			else:
				self.body += line

		self.processHeaders()

	@property
	def time(self):
		return self.meta['time'].strftime('%b %d %Y')

	@property
	def shortTime(self):
		return self.meta['time'].strftime('%d.%m.%y')

	@property
	def title(self):
		return self.meta['title']

	@property
	def slug(self):
		return self.meta['slug']

	@property
	def short(self):
		return markdown(self.body.split('\n\n\n')[0])

	@property
	def has_short(self):
		return (self.body.split('\n\n\n')[0].strip() != self.body.strip()) or 'ipython' in self.meta

	@property
	def url(self):
		return '/posts/' + self.slug + '.html'

	@property
	def tags(self):
		return self.meta['tags']

	def processHeaders(self):
		if 'tags' in self.meta:
			self.meta['tags'] = [x.strip() for x in self.meta['tags'].split(',')]
		else:
			self.meta['tags'] = []

		if 'time' in self.meta:
			self.meta['time'] = datetime.strptime(self.meta['time'], '%y-%m-%d %I:%M%p')

		self.hide = 'hide' in self.meta

	def render(self):
		body = self.body

		rendered = markdown(body)
		if 'ipython' in self.meta:
			content, info = ipynb.get_html_from_filepath(self.meta['ipython'])
			content = ipynb.fix_css(content, info)
			content = content.replace('imgs/', '/imgs/')
			rendered += content
		return rendered

	def __str__(self):
		return self.render()

def markdown(text):
	return pypandoc.convert_text(text, 'html', format='md')

def getPosts(hide):
	posts = []
	for post in os.listdir('posts'):
		if post.endswith('.md'):
			posts.append(Post(f'posts/{post}'))
	if hide:
		posts = list(filter(lambda p: not p.hide, posts))
	posts.sort(key=lambda x: x.meta['time'], reverse=True)
	return posts

def render_template(fn, **kwargs):
	return jinja2.Environment(
        loader=jinja2.FileSystemLoader('templates')
    ).get_template(fn).render(kwargs)

def genIndex(posts):
	# Create index
	indexes = [posts[i:i+POSTS_PER_PAGE] for i in range(0, len(posts), POSTS_PER_PAGE)]
	for i, ps in enumerate(indexes):
		fn = 'index.html' if i == 0 else f'index{i}.html'
		kwargs = {
			'blog' : ps
		}
		if i != 0:
			ind = i-1
			kwargs['future'] = '/' if ind == 0 else f'/index{ind}.html'
			kwargs['index'] = False
		if i != len(indexes) - 1:
			ind = i+1
			kwargs['past'] = '/' if ind == 0 else f'/index{ind}.html'
		kwargs['title'] = "wpovell"
		with open(os.path.join(OUT_DIR, fn), 'w') as f:
			f.write(render_template('index.html', **kwargs))

def genPosts(posts, ops):
	for post in posts:
		[op(post) for op in ops]
		fn = os.path.join(OUT_DIR, post.url[1:])
		with open(fn, 'w') as f:
			f.write(render_template('post.html', post=post, short=False, title=post.title))

def genTags(posts):
	tags = []
	for post in posts:
		tags.extend(post.tags)
	c = Counter(tags)

	with open(os.path.join(OUT_DIR, 'tags/index.html'), 'w') as f:
		f.write(render_template('tags.html', posts=posts, tags=c.most_common(), title="Tags"))

	for tag in set(tags):
		tagged_posts = filter(lambda x: tag in x.tags, posts)
		with open(os.path.join(OUT_DIR, f'tags/{tag}.html'), 'w') as f:
			f.write(render_template('tag.html', posts=tagged_posts, tag=tag, title=f"Tag: {tag}"))

def genAbout():
	with open(os.path.join(OUT_DIR, 'about.html'), 'w') as f:
		f.write(render_template('about.html', title="About"))

def cleanDist():
	try:
		dir_util.remove_tree(OUT_DIR)
	except FileNotFoundError:
		pass

def initDist():
	dir_util.mkpath(f'{OUT_DIR}/posts')
	dir_util.mkpath(f'{OUT_DIR}/tags')

import distutils.dir_util as dir_util

def copyStatic():
	dir_util.copy_tree('dev', OUT_DIR)

def sot(post):
	toInsert = '''<em>This post is a part of my
	<a href="/posts/summer-of-tensorflow.html">Summer of Tensorflow</a> series.</em>\n\n'''
	if 'SoT' in post.tags and post.slug != 'summer-of-tensorflow':
			post.body = toInsert + post.body


POSTS_PER_PAGE = 5
OUT_DIR = 'dist'
def main():
	from sys import argv

	hide = False
	if 'deploy' in argv:
		hide = True
		cleanDist()
	initDist()
	posts = getPosts(hide)
	ops = [sot]
	genIndex(posts)
	genAbout()
	genPosts(posts, ops)
	genTags(posts)
	copyStatic()

if __name__ == '__main__':
	main()