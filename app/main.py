import dataLoader as DL
import featureExtractor as FE
import models
import plotters

import numpy as np
from pprint import pprint

if __name__ == "__main__":
	trainVideos = 30

	#load trainSet
	#y_train: holds all valence values for each movie
	#x_train: holds all features for each movie
	#similar for test set
	(x_train, y_train, x_test, y_test) = DL.loadSinglePersonData(trainVideos)

	pprint(x_train)
	pprint(y_train)
	pprint(x_test)
	pprint(y_test)

	#models.linReg(x_train,y_train,x_test,y_test)
	models.linReg(x_train,y_train,x_test,y_test)

	#plotters.plot3D(x_train[0], x_train[1], y_train, 'leftPower', 'rightPower', 'Valence' )