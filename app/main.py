import dataLoader as DL
import featureExtractor as FE
import models
import plotters

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from pprint import pprint

def regression():
	trainVideos = 30

	#load trainSet
	#y_train: holds all valence values for each movie
	#x_train: holds all features for each movie
	#similar for test set
	(x_train, y_train, x_test, y_test) = DL.loadSinglePersonData(trainVideos)

	#models.linReg(x_train,y_train,x_test,y_test)
	models.ridgeReg(x_train,y_train,x_test,y_test)

	#plotters.plot3D(x_train[0], x_train[1], y_train, 'leftPower', 'rightPower', 'Valence' )

if __name__ == "__main__":
	regression()