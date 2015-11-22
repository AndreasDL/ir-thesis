import os
import pickle
import numpy as np
import featureExtractor as FE
from sklearn.cross_validation import train_test_split
import random

def loadSinglePersonData(person, borderDist= 0.125, classCount =  2, pad='../dataset', featureFunc=FE.extract):
	if 1 <= borderDist*2:
		print('borderDist (', borderDist, ' is too large for the given number of classes (', classCount, ')')
		exit(-1)

	X, y = [], []

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

		valences = np.array( data['labels'][:,0] ) #ATM only valence needed
		valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1

		#transform into classes
		valences = valences * classCount
		classes = np.floor(valences)
		classes[ valences == classCount] = classCount - 1

		used_indexes = []
		for i, valence, klass in zip(range(len(data['labels'])),valences, classes):
			
			#distance to borders
			dist_up, dist_bot = 1, 1 #classwidth is always 1
			if klass < classCount - 1:  #if there is an upper border
				dist_up  = (klass+1) - valence #distance to upper border
			elif klass > 0: #if there is a bottom border
				dist_bot = valence - klass #distance to bottom border
			

			if dist_up > borderDist and dist_bot > borderDist: #add if sample is far enough from border
				y.append(klass)
				used_indexes.append(i)

		#extract features
		for i in used_indexes:
			X.append( featureFunc(data['data'][i]) )
	
	return [np.array(X), np.array(y)]



def loadSinglePersonData_old(person, test_chance, pad='../dataset', happyThres=0.5, borderDist=0.125, featureFunc=FE.extract, hardTest=True):
	#loads data and creates two classes happy and not happy

	X_train, X_test = [], []
	y_train, y_test = [], []

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

		valences = np.array( data['labels'][:,0] ) #ATM only valence needed
		valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1

		#only keep elements that are suffiently positioned away from the border		
		underbound = happyThres - borderDist
		upperbound = happyThres + borderDist

		used_indexes_train = []
		used_indexes_test  = []
		
		for index, valence in enumerate(valences):
			if random.random() < test_chance:
				if hardTest:
					used_indexes_test.append(index)
					if valence <= happyThres:
						y_test.append(0)
					else:
						y_test.append(1)

				else: #no hard test => only use clear examples in test set
					if valence <= underbound: #sad
						y_test.append(0)
						used_indexes_test.append(index)

					elif valence >= upperbound: #happy
						y_test.append(1)
						used_indexes_test.append(index)

			else:
				#trainset should only hold clear examples
				if valence <= underbound: #sad
					y_train.append(0)
					used_indexes_train.append(index)

				elif valence >= upperbound: #happy
					y_train.append(1)
					used_indexes_train.append(index)

		#extract features
		for i in used_indexes_train: #for each video
			X_train.append( featureFunc(data['data'][i]) )
		for i in used_indexes_test:
			X_test.append( featureFunc(data['data'][i]) )


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