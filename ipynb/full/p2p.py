import glob
import os
from PIL import Image

def data_generator(path, batch_size=1):
	paths = glob.glob(os.path.join(path, '*.jpg'))
	toYield = []
	for path in paths:
		im = Image.open(path)
		arr = np.asarray(im, dtype=np.float32)
		toYield.append(arr / 255 * 2 - 1)
		im.close()
	while True:
		for arr in toYield:
			yield arr


train = data_generator('/Users/data/pix2pix/facades/train')
test = data_generator('/Users/data/pix2pix/facades/val')
for i in range(10):
