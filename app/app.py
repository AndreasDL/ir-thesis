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
			#structure
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
	#fraction of activity at right side
	#todo L-R/L+R voor alpha components zie gegeven paper p6
	sum_right = 0
	sum_left = 0

	sum_left += np.sum(samples[1,:])
	sum_left += np.sum(samples[3,:])
	sum_left += np.sum(samples[4,:])
	sum_left += np.sum(samples[7,:])
	sum_left += np.sum(samples[8,:])
	sum_left += np.sum(samples[11,:])
	sum_left += np.sum(samples[14,:])
	
	sum_right += np.sum(samples[16,:])
	sum_right += np.sum(samples[19,:])
	sum_right += np.sum(samples[21,:])
	sum_right += np.sum(samples[24,:])
	sum_right += np.sum(samples[26,:])
	sum_right += np.sum(samples[28,:])
	sum_right += np.sum(samples[30,:])
	sum_right += np.sum(samples[31,:])

	#fraction of activity at right side
	return [ sum_right/( sum_right + sum_left + 1 ) ]

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

	Fs = 150                         # sampling rate
	Ts = 1.0/Fs                      # sampling interval
	t = np.arange(0,1,Ts)            # time vector
	ff = 5                           # frequency of the signal
	y = np.sin(2 * np.pi * ff * t)

	plt.subplot(2,1,1)
	plt.plot(t,y,'k-')
	plt.xlabel('time')
	plt.ylabel('amplitude')

	plt.subplot(2,1,2)
	n = len(y)                       # length of the signal
	k = np.arange(n)
	T = n/Fs
	frq = k/T # two sides frequency range
	freq = frq[range(round(n/2))]           # one side frequency range

	Y = np.fft.fft(y)/n              # fft computing and normalization
	Y = Y[range(round(n/2))]

	plt.plot(freq, abs(Y), 'r-')
	plt.xlabel('freq (Hz)')
	plt.ylabel('|Y(freq)|')

	plt.show()