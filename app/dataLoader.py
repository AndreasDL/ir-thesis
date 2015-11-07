import os
import pickle
import numpy as np
from featureExtractor import extract
from sklearn.cross_validation import train_test_split

def loadSinglePersonData(person, test_size=4, pad='../dataset', happyThres=0.5):
	#loads data and creates two classes happy and not happy

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

		y = np.array( data['labels'][:,0] ) #ATM only valence needed
		y = (y - 1) / 8 #1->9 to 0->8 to 0->1
		y[ y <= happyThres ] = 0
		y[ y > happyThres ] = 1

		#extract features
		for j in range(len(data['data'])): #for each video
			X.append( extract(data['data'][j]) )

	#split into train and test set, while shuffeling
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=42)

	return [np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)]

def loadMultiplePersonsData(personCount=32, test_size=8, pad='../dataset', happyThres=0.5):
	#loads data for all persons, for each person put part of videos in test set
	#create two classes happy and not happy
	
	X_train, y_train = [], []
	X_test , y_test  = [], []

	for person in range(1,personCount+1):
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

			y = np.array( data['labels'][:,0] ) #ATM only valence needed
			y = (y - 1) / 8 #1->9 to 0->1
			y[ y <= happyThres ] = 0
			y[ y > happyThres ] = 1

			#extract features
			X = []
			for j in range(len(data['data'])): #for each video
				X.append( extract(data['data'][j]) )

		#split into train and test set, while shuffeling
		X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=test_size, random_state=42)

		#add to list
		X_train.extend(X_tr)
		X_test.extend(X_te)
		
		y_train.extend(y_tr)
		y_test.extend(y_te)


	return [np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)]

def dump(X_train, y_train, X_test, y_test, name, path='../dumpedData'):
	fname = path + '/' + name
	data = { 'X_train': X_train, 
		'y_train': y_train,
		'X_test': X_test,
		'y_test': y_test 
	}
	with open(fname, 'wb') as f:
		pickle.dump( data, f )

def load(name, path='../dumpedData'):
	fname = path + '/' + name

	data = None
	with open(fname,'rb') as f:
		p = pickle._Unpickler(f)
		p.encoding= ('latin1')
		data = p.load()
	if data == None:
		print('data loading failed for file:', fname)
		exit(-1)

	return data['X_train'], data['y_train'], data['X_test'], data['y_test']