from datetime import datetime

from gen.util import toHTML
import gen.ipynb as ipynb

class Post:
    def __init__(self, file):
        self.file = file
        with open(file, encoding='utf-8') as f:
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
        return self.meta['time'].strftime('%d.%m.%y')

    @property
    def title(self):
        return self.meta['title']

    @property
    def slug(self):
        return self.file.stem

    @property
    def tags(self):
        return self.meta['tags']

    @property
    def url(self):
        return '/posts/' + self.slug + '.html'

    def processHeaders(self):
        if 'tags' in self.meta:
            self.meta['tags'] = [x.strip() for x in self.meta['tags'].split(',')]
        else:
            self.meta['tags'] = []

        if 'time' in self.meta:
            self.meta['time'] = datetime.strptime(self.meta['time'], '%y-%m-%d')

        self.hide = 'hide' in self.meta

    def render(self):
        body = self.body
        rendered = toHTML(body, self.file.suffix[1:])
        if 'ipython' in self.meta:
            content, info = ipynb.get_html_from_filepath(self.meta['ipython'])
            content = ipynb.fix_css(content, info)
            content = content.replace('imgs/', '/imgs/')
            rendered += content

        return rendered

    def __str__(self):
        return self.render()
