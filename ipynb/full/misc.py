import json
import os

def showImg(im, title):
    ''' Display image with title. '''	
	try:
		plt
	except NameError:
		from matplotlib import pyplot as plt
    plt.title(title)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.show()

class DataSaver:
	def __init__(self, name):
		self.name = name
		self.data = {}
		self.outFn = f'out/{name}/data.json'
		if os.path.exists(self.outFn):
			with open(self.outFn) as f:
				self.data = json.load(f)

	def add(self, name, value, step):
		if not name in self.data:
			self.data[name] = {}
		step = int(step)
		self.data[name][step] = value

	def save(self):
		with open(self.outFn, 'w') as f:
			json.dump(self.data, f)

	def graph(self, names, AVG_WIN = 100):
		if isinstance(names, str):
			names = [names]
		try:
			plt
		except Exception:
			from matplotlib import pyplot as plt

		for name in names:
			x, y = zip(*self.data[name].items())
			if AVG_WIN > 1:
				win = AVG_WIN//2
				newY = []
				for i in range(win, len(y)-win):
					l = y[i - win : i + win]
					newY.append(sum(l) / len(l))		
				y = newY
				x = list(map(lambda x: x + win, range(len(y))))
			plt.plot(x,y, label=name)
		if len(names) > 1:
			plt.legend()
		plt.show()

