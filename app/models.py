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

	# The coefficients
	print('model: linear\n\ttrain error: ' , np.mean((regr.predict(x_train) - y_train) ** 2), '\n\tcoef:' , regr.coef_)

	# Plot outputs
	plt.scatter(x_train, y_train,  color='black')
	plt.plot(x_train, regr.predict(x_train), color='blue', linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.xlabel('metric')
	plt.ylabel('valence|')
	
	plt.show()
	return

def ridgeReg(x_train,y_train,x_test,y_test):
	#perform linear regression
	alphaValues = [0,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]
	cvSets = 8
	cvSize = len(x_train) / cvSets

	#get sets
	err = np.zeros(len(alphaValues))
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

			err[i] += np.mean((regr.predict(x_cv) - y_cv) ** 2)
	err /= cvSets
	pprint(err)
	print('model: ridge\n\ttrain error: ' , err[np.argmin(err)], '\n\talpha: ', alphaValues[np.argmin(err)], '\n\tcoef:' , regr.coef_)

	# Plot outputs
	#TODO testsets!!
	regr = SKL.linear_model.Ridge(alpha=alphaValues[np.argmin(err)]) #	Create linear regression object
	regr.fit(x_train, y_train) # Train the model using the training sets

	plt.scatter(x_train, y_train,  color='black')
	plt.plot(x_train, regr.predict(x_train), color='blue',linewidth=3)
	plt.xticks(())
	plt.yticks(())
	plt.xlabel('metric')
	plt.ylabel('valence')
	plt.show()
	
	return

def polyReg(x_train,y_train,x_test,y_test):

	model = Pipeline([('poly', PolynomialFeatures(degree=3)),
		('linear', LinearRegression(fit_intercept=False))])
	
	# fit to an order-3 polynomial data
	model = model.fit(x_train, y_train)

	# Plot outputs
	plt.scatter(x_test, y_test,  color='black')
	plt.plot(x_test, model.predict(x_test), color='blue',
	         linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.xlabel('metric')
	plt.ylabel('valence|')

	plt.show()
	return 