import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn as SKL
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from pprint import pprint

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def linReg(x_train,y_train,x_test,y_test):
	#perform linear regression
	
	#Create linear regression object
	regr = SKL.linear_model.LinearRegression(n_jobs=-1)

	# Train the model using the training sets
	regr.fit(x_train, y_train)
	
	train_err = np.mean((regr.predict(x_train) - y_train) ** 2)
	test_err  = np.mean((regr.predict(x_test)  - y_test) ** 2)

	return train_err, test_err, regr
def ridgeReg(x_train,y_train,x_test,y_test, cvSets= 8):
	#perform linear regression
	alphaValues = [0,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]
	cvSize = round( len(x_train) / cvSets )

	#get sets
	err   = np.zeros( len(alphaValues) )
	regrs = []
	for j in range(cvSets):
		x = np.concatenate( (x_train[:cvSize * j], x_train[cvSize * (j+1):]), 0 )
		y = np.concatenate( (y_train[:cvSize * j], y_train[cvSize * (j+1):]), 0 )

		x_cv = x_train[cvSize * j: cvSize * (j+1)]
		y_cv = y_train[cvSize * j: cvSize * (j+1)]

		#check the different alpha values
		for i in range(len(alphaValues)):
			alpha = alphaValues[i]

			regr = SKL.linear_model.Ridge(alpha=alpha) #Create linear regression object
			regr.fit(x, y) # Train the model using the training sets

			err[i] += np.mean( (regr.predict(x_cv) - y_cv)** 2 )
			regrs.append(regr)
	
	train_err = err[np.argmin(err)]
	test_err  = np.mean( (regr.predict(x_test) - y_test)**2 )
	regr = regrs[np.argmin(err)]

	return train_err, test_err, regr