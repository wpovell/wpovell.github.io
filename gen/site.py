from pathlib import Path
import shutil
import distutils.dir_util as dir_util
from datetime import datetime as dt
from jinja2 import Environment, FileSystemLoader, select_autoescape

from gen.post import Post

class Site:
    def __init__(self, deploy, out='dist', posts='posts', static='static'):
        self.out_dir = Path(out)
        self.posts_dir = Path(posts)
        self.static_dir = Path(static)

        self.time = dt.now().strftime("%Y.%m.%d")

        self.deploy = deploy
        self.env = Environment(
            loader=FileSystemLoader('templates'),
            autoescape=select_autoescape(['html'])
        )

    def initDist(self):
        if self.deploy:
            shutil.rmtree(self.out_dir)
        Path.mkdir(self.out_dir / 'posts', parents=True, exist_ok=True)


    def render_template(self, fn, **kwargs):
        kwargs['updated'] = self.time
        return self.env.get_template(fn).render(kwargs)

    def genPosts(self):
        posts = []
        for p in self.posts_dir.iterdir():
            post = Post.from_dir(p)

            if post is None or (self.deploy and post.hide):
                continue

            post.create(self.out_dir / 'posts', self.render_template)
            posts.append(post)

        posts.sort(key=lambda p: p.meta['time'], reverse=True)
        return posts

    def genIndex(self, posts):
        with open(self.out_dir / 'index.html', 'w') as f:
            f.write(self.render_template('index.html', posts=posts))

    def genTags(self):
        pass

    def copyStatic(self):
        dir_util.copy_tree(str(self.static_dir), str(self.out_dir))

    def render(self):
        self.initDist()
        posts = self.genPosts()
        self.genIndex(posts)
        self.genTags()
        self.copyStatic()