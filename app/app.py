import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from sklearn import linear_model

def loadData(pad, startFileIndex, fileCount):
	features = np.zeros((40, 1)) #40 x 1 feature
	labels = np.zeros((40, 1)) #40 x 1 label

	#load testSet
	for i in range(fileCount):
		fname = 'dataset/s'
		if i+1+startFileIndex < 10:
			fname += '0' 
		fname += str(i+1+startFileIndex) + '.dat'
		
		with open(fname,'rb') as f:
			p = pickle._Unpickler(f)
			p.encoding= ('latin1')
			data = p.load()
			#structure of data element:
			#data['labels'][video , attribute]
			#data['data'][video, channel, value]

			labels = data['labels'][:,1] #only valence needed
			#structure y_train[video]

			#get features
			#different left/right right more active than left
			#calculate sum_left & sum_right & sum_total = sum_left + sum_right
			#use feature= sum_right / sum_total
			#don't use sum(x['data']), cuz we ignore center electrodes

			for j in range(len(data['data'])): #for each video
				features[j] = calculateFeatures(data['data'][j])


	return [features, labels]

def calculateFeatures(samples):
	#structure of samples[channel, sample]

	#FFT to get frequency components
	Fs = 128
	n = len(samples[0])
	alpha_component = np.zeros(40)
	for i in [1,3,4,7,8,11,14,16,19,21,24,26,28,30,31]:
		Y = np.fft.fft(samples[i])/n 					# fft computing and normalization
		Y = Y[range(round(n/2))]
		
	#sum Alpha components for left and right
	#alpha runs from 8 - 13Hz
	#freq = index * Fs / n => value = abs(Y[j])
	#8hz = index * Fs/n <=> index = 8hz * 4032 / 128
	startIndex  = round(8  * n / Fs)
	stopIndex   = round(13 * (n / Fs))
	alpha_left  = 0
	alpha_right = 0
	for i in [1,3,4,7,8,11,14]:
		for j in range(startIndex,stopIndex+1):
			alpha_left += Y[j]

	for i in [16,19,21,24,26,28,30,31]:
		for j in range(startIndex,stopIndex+1):
			alpha_right += Y[j]		

	return [ (alpha_left - alpha_right) / (alpha_left + alpha_right) ]
	#L-R/L+R voor alpha components zie gegeven paper p6


if __name__ == "__main__":
	trainPersons = 1 # one set = one person
	testPersons = 1 # one set = one person

	#load trainSet
	(x_train, y_train) = loadData('dataset/s', 0, trainPersons)

	#load testSet
	(x_test, y_test) = loadData('dataset/s', trainPersons, testPersons)

	#starting from here we have
	#a trainset consisting of trainPersons 
	#y_train: holds all the valence values for each movie
	#x_train: holds all the right fraction values for each movie
	#
	#a testset consisting of testPersons
	#y_train: holds all valence values for each movie
	#x_train: holds all the right fraction values for each movie

	#starting here we will
	#perform linear regression
	# Create linear regression object
	regr = linear_model.LinearRegression()

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