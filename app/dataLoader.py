import pickle
import numpy as np
from sklearn import linear_model
import featureExtractor as FE

from pprint import pprint



def loadMultiPersonData(train_fileCount, test_fileCount, pad='dataset/s'):
	x_train = np.zeros((40, 1)) #40 x 1 feature
	y_train = np.zeros((40, 1)) #40 x 1 label
	x_test  = np.zeros((40, 1)) #40 x 1 feature
	y_test  = np.zeros((40, 1)) #40 x 1 label

	#load trainSet
	for i in range(train_fileCount):
		fname = pad
		if i+1 < 10:
			fname += '0' 
		fname += str(i+1) + '.dat'
		
		with open(fname,'rb') as f:
			p = pickle._Unpickler(f)
			p.encoding= ('latin1')
			data = p.load()
			#structure of data element:
			#data['labels'][video , attribute]
			#data['data'][video, channel, value]

			y_train = data['labels'][:,1] #only valence needed
			#structure y_train[video]

			for j in range(len(data['data'])): #for each video
				x_train[j] = FE.calculateFeatures(data['data'][j])

	#load testSet
	for i in range(test_fileCount):
		fname = pad
		if i+1+train_fileCount < 10:
			fname += '0' 
		fname += str(i+1+train_fileCount) + '.dat'
		
		with open(fname,'rb') as f:
			p = pickle._Unpickler(f)
			p.encoding= ('latin1')
			data = p.load()
			#structure of data element:
			#data['labels'][video , attribute]
			#data['data'][video, channel, value]

			y_test = data['labels'][:,1] #only valence needed

			for j in range(len(data['data'])): #for each video
				x_test[j] = FE.calculateFeatures(data['data'][j])

	return [x_train, y_train, x_test, y_test]

def loadSinglePersonData(trainVideoCount, pad='dataset/s'):
	x_train = np.zeros((trainVideoCount, 1)) # .. x 1 feature
	y_train = np.zeros((trainVideoCount, 1)) # .. x 1 label
	x_test  = np.zeros((40-trainVideoCount, 1)) # .. x 1 feature
	y_test  = np.zeros((40-trainVideoCount, 1)) # .. x 1 label

	fname = str(pad) +'01.dat'
	with open(fname,'rb') as f:
		p = pickle._Unpickler(f)
		p.encoding= ('latin1')
		data = p.load()
		#structure of data element:
		#data['labels'][video , attribute]
		#data['data'][video, channel, value]

		y_train = data['labels'][0:trainVideoCount,1] #only valence needed
		y_test  = data['labels'][trainVideoCount:,1]

		#split single person in test and train set
		for j in range(trainVideoCount): #for each video
			x_train[j] = FE.calculateFeatures(data['data'][j])
		
		for j in range(len(data['data']) - trainVideoCount): #for each video
			x_test[j] = FE.calculateFeatures(data['data'][j])

	return [x_train, y_train, x_test, y_test]