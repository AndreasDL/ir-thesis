import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn as SKL
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def linReg(x_train,y_train,x_test,y_test):
	#perform linear regression
	
	#Create linear regression object
	regr = SKL.linear_model.LinearRegression(normalize=True, copy_X=False,n_jobs=-1)

	# Train the model using the training sets
	regr.fit(x_train, y_train)

	# The coefficients
	print('Coefficients: \n', regr.coef_)

	# The mean square error
	print("Residual sum of squares: %.2f"
	      % np.mean((regr.predict(x_test) - y_test) ** 2))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % regr.score(x_test, y_test))

	# Plot outputs
	plt.scatter(x_test, y_test,  color='black')
	plt.plot(x_test, regr.predict(x_test), color='blue',
	         linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.xlabel('metric')
	plt.ylabel('valence|')
	
	plt.show()
	return

def ridgeReg(x_train,y_train,x_test,y_test):
	#perform linear regression
	
	#Create linear regression object
	regr = SKL.linear_model.Ridge(normalize=True, copy_X=False,alpha=0.5)

	# Train the model using the training sets
	regr.fit(x_train, y_train)

	# The coefficients
	print('Coefficients: \n', regr.coef_)

	# The mean square error
	print("Residual sum of squares: %.2f"
	      % np.mean((regr.predict(x_test) - y_test) ** 2))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % regr.score(x_test, y_test))

	# Plot outputs
	plt.scatter(x_test, y_test,  color='black')
	plt.plot(x_test, regr.predict(x_test), color='blue',
	         linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.xlabel('metric')
	plt.ylabel('valence|')
	
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