import pickle
import numpy as np
from featureExtractor import extract
from sklearn.cross_validation import train_test_split

def loadSinglePersonData(person, test_size=4, pad='../dataset'):

	X = []
	y = None

	fname = str(pad) + '/s'
	if person < 10:
		fname += '0'
	fname += str(person) + '.dat'
	with open(fname,'rb') as f:
		p = pickle._Unpickler(f)
		p.encoding= ('latin1')
		data = p.load()
		#structure of data element:
		#data['labels'][video] = [valence, arousal, dominance, liking]
		#data['data'][video][channel] = [samples * 8064]

		y = data['labels'][:,0] #only valence needed

		#extract features
		for j in range(len(data['data'])): #for each video
			X.append( extract(data['data'][j]) )

	#split into train and test set, while shuffeling
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=42)

	return [np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)]

#deprecated
def loadMultiPersonData(train_fileCount, test_fileCount, pad='../dataset/s'):
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

			y_train = data['labels'][:,0] #only valence needed
			#structure y_train[video]

			for j in range(len(data['data'])): #for each video
				x_train[j] = extract(data['data'][j])

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
				x_test[j] = extract(data['data'][j])

	return [x_train, y_train, x_test, y_test]