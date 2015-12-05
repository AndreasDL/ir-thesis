import os
import pickle
import numpy as np
import featureExtractor as FE
from sklearn.cross_validation import train_test_split
import random

def loadPerson(person, featureFunc, csp, use_median, pad='../dataset'):
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

        #only use EEG channels
        samples = np.array(data['data'])[:,:32,:] #throw out non-EEG channels

        #rescale
        valences = np.array( data['labels'][:,0] ) #ATM only valence needed
        valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1
        
        if use_median:
            #median as border
            border = np.median(valences)
        else:
            border = 0.5
        
        #classes
        y = valences
        y[ y <= border ] = 0
        y[ y >  border ] = 1
        y = np.array(y, dtype='int')

        #preprocessing
        X_prepped, filters = csp.csp(samples=samples,labels=y)
        
        #extract features for each video
        for video in range(len(data['data'])):
            X.append( featureFunc(X_prepped[video]) )

    return [np.array(X), y]

def loadSinglePersonData_old(person, test_chance, featureFunc, pad='../dataset', happyThres=0.5, borderDist=0.125, hardTest=True):
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