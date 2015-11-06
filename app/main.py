import dataLoader as DL
import models
import plotters

if __name__ == "__main__":
	test_size = 4
	used_person = 7

	#load trainSet
	#y_train: holds all valence values for each movie
	#x_train: holds all features for each movie
	#similar for test set
	(X_train, y_train, X_test, y_test) = DL.loadSinglePersonData(person=used_person, test_size=test_size)

	#linear regression
	train_err, test_err, regr = models.linReg(X_train,y_train,X_test,y_test)
	print('model: linear',
		'\n\tTrain error: ', train_err,
		'\n\tTest error: ' , test_err
	)

	#ridge regression
	train_err, test_err, regr = models.ridgeReg(X_train, y_train, X_test, y_test, cvSets=2)
	print('model: ridge',
		'\n\tTrain error: ', train_err,
		'\n\tTest error: ' , test_err
	)