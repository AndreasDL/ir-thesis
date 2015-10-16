import dataLoader as DL
import featureExtractor as FE
import models
import plotters

import numpy as np
from pprint import pprint

if __name__ == "__main__":
	trainPersons = 32 # one set = one person
	testPersons = 0 # one set = one person

	#load trainSet
	#y_train: holds all valence values for each movie
	#x_train: holds all features for each movie
	(x_train, y_train) = DL.loadData('dataset/s', 0, trainPersons)

	#load testSet
	#y_test: holds all the valence values for each movie
	#x_test: holds all features for each movie
	(x_test, y_test) = DL.loadData('dataset/s', trainPersons, testPersons)	
		
	x_train = np.transpose(x_train)
	plotters.plot3D(x_train[0], x_train[1], y_train, 'leftPower', 'rightPower', 'Valence' )