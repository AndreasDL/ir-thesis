import dataLoader as DL
import featureExtractor as FE
import models
import plotters

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
	trainVideos = 32

	#load trainSet
	#y_train: holds all valence values for each movie
	#x_train: holds all features for each movie
	#similar for test set
	(x_train, y_train, x_test, y_test) = DL.loadSinglePersonData(7,trainVideos)

	#linear regression
	train_err, test_err, regr = models.linReg(x_train,y_train,x_test,y_test)
	print('model: linear',
		'\n\tTrain error: ', train_err,
		'\n\tTest error: ' , test_err
	)

	#ridge regression
	train_err, test_err, regr = models.ridgeReg(x_train, y_train, x_test, y_test, cvSets=2)
	print('model: ridge',
		'\n\tTrain error: ', train_err,
		'\n\tTest error: ' , test_err
	)