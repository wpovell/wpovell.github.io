from datetime import datetime
import shutil
from pathlib import Path
import re
import yaml
from gen.util import toHTML

class Post:
    @staticmethod
    def from_dir(post):
        md = list(post.glob("*.md"))
        if len(md) == 0:
            return None

        return Post(md[0], post)


    def __init__(self, file, root):
        self.file = file
        self.root = root
        with open(file, encoding='utf-8') as f:
            data=f.read()

        header=re.compile(r'---.*---', re.DOTALL)
        m=header.search(data)
        self.meta = yaml.load(data[4:m.span()[1]-4])
        self.body = data[m.span()[1]:]

    @property
    def time(self):
        return self.meta['time'].strftime('%d.%m.%y')

    @property
    def title(self):
        return self.meta['title']

    @property
    def slug(self):
        return self.root.stem

    @property
    def tags(self):
        return self.meta['tags']

    @property
    def url(self):
        return '/posts/' + self.slug

    @property
    def hide(self):
        return 'hide' in self.meta

    def render(self, short=False):
        if short:
            body = self.body.split("\n\n")[0]
        else:
            body = self.body

        rendered = toHTML(body, self.file.suffix[1:])
        return rendered

    def create(self, out, render):
        out = out / self.slug

        if out.exists():
            shutil.rmtree(str(out))

        shutil.copytree(str(self.root), str(out))

        with open(out / 'index.html', 'w') as f:
            f.write(render('post.html', post=self))

    def __str__(self):
        return self.render()
