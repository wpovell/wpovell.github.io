import pypandoc

def toHTML(text, format='md'):
    return pypandoc.convert_text(text, 'html', format=format)
