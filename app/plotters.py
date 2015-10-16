import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot2D(x,y, labelX='X label', labelY='Y label'):
	plt.scatter(x, y,  color='black')

	plt.xlabel(labelX)
	plt.ylabel(labelY)

	plt.xticks(())
	plt.yticks(())

	plt.show()


def plot3D(x,y,z, labelX='X label', labelY='Y label', labelZ='Z label'):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z, c='r')

	ax.set_xlabel(labelX)
	ax.set_ylabel(labelY)
	ax.set_zlabel(labelZ)

	plt.show()