import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

def linReg(x_train,y_train,x_test,y_test):
	#starting here we will
	#perform linear regression
	# Create linear regression object
	regr = linear_model.LinearRegression(normalize=True)

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