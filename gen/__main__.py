from gen.site import Site
from sys import argv
site = Site('deploy' in argv)
site.render()
