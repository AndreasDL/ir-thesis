import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pprint import pprint

def plot2D(x,y, labelX='X label', labelY='Y label', title=''):
	plt.plot(x, y, '-o')# color='black')

	plt.xlabel(labelX)
	plt.ylabel(labelY)

	plt.xticks(())
	plt.yticks(())
	
	plt.show()

def plot2DSub(x,y,labelsX,labelY):
	# Four axes, returned as a 2-d array
	f, axarr = plt.subplots(round(len(labelsX) / 2), 2)

	for i in range(0,len(x[0]),2):
		axarr[np.floor(i/2), 0].scatter(x[:,i], y)
		axarr[np.floor(i/2), 0].set_title(labelsX[i])
		
		axarr[np.floor(i/2), 1].scatter(x[:,i+1], y)
		axarr[np.floor(i/2), 1].set_title(labelsX[i+1])

	# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
	plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
	plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

	plt.show()

def plot3D(x,y,z, labelX='X label', labelY='Y label', labelZ='Z label'):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z, c='r')

	ax.set_xlabel(labelX)
	ax.set_ylabel(labelY)
	ax.set_zlabel(labelZ)

	plt.show()
