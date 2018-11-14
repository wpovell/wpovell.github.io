from pathlib import Path
import shutil
import distutils.dir_util as dir_util

import jinja2

from gen.post import Post

OUT_DIR = Path('dist')
POST_DIR = Path('posts')
STATIC_DIR = Path('static')

def initDist(clean):
    if clean:
        shutil.rmtree(OUT_DIR)
    Path.mkdir(OUT_DIR / 'posts', parents=True, exist_ok=True)

def render_template(fn, **kwargs):
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader('templates')
    ).get_template(fn).render(kwargs)

def genPosts(hide):
    posts = []
    for p in POST_DIR.iterdir():
        if not p.is_file():
            continue
        post = Post(p)
        posts.append(post)

    posts.sort(key=lambda x: x.meta['time'], reverse=True)
    for post in posts:
        fn = OUT_DIR / 'posts' / (post.slug + '.html')
        with open(fn, 'w') as f:
            f.write(render_template('post.html', post=post, hide=hide))
    return posts

def genIndex(posts):
    with open(OUT_DIR / 'index.html', 'w') as f:
        f.write(render_template('index.html', posts=posts))

def genTags():
    pass

def copyStatic():
    dir_util.copy_tree(str(STATIC_DIR), str(OUT_DIR))

def main():
    from sys import argv

    hide = False
    clean = False
    if 'deploy' in argv:
        hide = True
        clean = True
    initDist(clean)
    posts = genPosts(hide)
    genIndex(posts)
    # genTags()
    copyStatic()

if __name__ == '__main__':
    main()
