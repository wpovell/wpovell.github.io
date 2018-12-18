import pypandoc

def to_HTML(text, format='md'):
    args = [
        "--highlight-style=misc/code.theme",
        "--standalone",
        "--mathjax",
    ]
    return pypandoc.convert_text(text, 'html5', extra_args=args, format=format)
