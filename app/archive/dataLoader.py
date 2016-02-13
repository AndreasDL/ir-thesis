import pickle
import random

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from archive import featureExtractor as FE
from archive.csp import Csp

channelNames = {
    'Fp1' : 0 , 'AF3' : 1 , 'F3'  : 2 , 'F7'  : 3 , 'FC5' : 4 , 'FC1' : 5 , 'C3'  : 6 , 'T7'  : 7 , 'CP5' : 8 , 'CP1' : 9, 
    'P3'  : 10, 'P7'  : 11, 'PO3' : 12, 'O1'  : 13, 'Oz'  : 14, 'Pz'  : 15, 'Fp2' : 16, 'AF4' : 17, 'Fz'  : 18, 'F4'  : 19,
    'F8'  : 20, 'FC6' : 21, 'FC2' : 22, 'Cz'  : 23, 'C4'  : 24, 'T8'  : 25, 'CP6' : 26, 'CP2' : 27, 'P4'  : 28, 'P8'  : 29,
    'PO4' : 30, 'O2'  : 31, 
    'hEOG' : 32, #(horizontal EOG:  hEOG1 - hEOG2)    
    'vEOG' : 33, #(vertical EOG:  vEOG1 - vEOG2)
    'zEMG' : 34, #(Zygomaticus Major EMG:  zEMG1 - zEMG2)
    'tEMG' : 35, #(Trapezius EMG:  tEMG1 - tEMG2)
    'GSR'  : 36, #(values from Twente converted to Geneva format (Ohm))
    'Respiration belt' : 37,
    'Plethysmograph' : 38,
    'Temperature' : 39
}

def loadPerson(person, featureFunc, prefilter=False, use_median=False, use_csp=True, pad='../dataset'):
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

        if prefilter:
            #get alpha power only, before applying CSP
            samples = FE.filterBand(samples, 'alpha')

        #rescale
        valences = np.array( data['labels'][:,0] ) #ATM only valence needed
        valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1

        #determine border
        if use_median:
            #median as border
            border = np.median(valences)
        else:
            border = 0.5
        
        #assign classes
        y = valences
        y[ y <= border ] = 0
        y[ y >  border ] = 1
        y = np.array(y, dtype='int')

        #break off test set
        sss = StratifiedShuffleSplit(y, n_iter=10, test_size=0.25, random_state=19)
        for train_set_index, test_set_index in sss:
            X_train, y_train = samples[train_set_index], y[train_set_index]
            X_test , y_test  = samples[test_set_index] , y[test_set_index]
            break;  #abuse the shuffle split, we want a static break, not a crossvalidation

        #preprocessing with CSP (16 channelPairs)
        csp=None
        if use_csp:
            #fit csp to train set
            csp = Csp(samples=X_train,labels=y_train)

            #transform train and testset
            X_train = csp.apply_all(X_train)
            X_test  = csp.apply_all(X_test)

        #extract features
        feat_X_train = []
        feat_X_test  = []
        for i in range(len(X_train)):
            feat_X_train.append( featureFunc(X_train[i]) )
        
        for i in range(len(X_test)):
            feat_X_test.append(  featureFunc(X_test[i] ) )

    return [np.array(feat_X_train), np.array(y_train), np.array(feat_X_test), np.array(y_test), csp]

def loadPersonEpochDimRedu(person, featureFunc, pad='../dataset'):
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
        '''
        all_samples = data['data']#
        samples = []
        samples.append(all_samples[:,channelNames['F3'],:])
        samples.append(all_samples[:,channelNames['F4'],:])
        samples.append(all_samples[:,channelNames['F7'],:])
        samples.append(all_samples[:,channelNames['F8'],:])
        samples.append(all_samples[:,channelNames['T7'],:])
        samples.append(all_samples[:,channelNames['T8'],:])
        samples.append(all_samples[:,channelNames['AF3'],:])
        samples.append(all_samples[:,channelNames['AF4'],:])
        samples = np.array(samples)'''

        #rescale
        valences = np.array( data['labels'][:,0] ) #ATM only valence needed
        valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1

        arousals = np.array( data['labels'][:,1] )
        arousals = (arousals - 1 ) / 8
        
        #classes
        val_labels = valences
        val_labels[ val_labels <= 0.5 ] = 0
        val_labels[ val_labels >  0.5 ] = 1
        
        arr_labels = arousals
        arr_labels[ arr_labels <= 0.5 ] = 0
        arr_labels[ arr_labels >  0.5 ] = 1

        labels = []
        for val, arr in zip(val_labels, arr_labels):
            labels.append( 2*val + arr)

        labels = np.array(labels, dtype='int')

        #epochs
        Fs = 128
        epochlength = 6 * Fs
        epochoverlap = 5 * Fs
        for video, label in zip(samples, labels):
            for i in range(0,len(samples[0][0]), epochlength - epochoverlap):
                
                epoch = video[:, i:i+epochlength]
                features = featureFunc(epoch)
                
                X.append(features)
                y.append(label)

        y = np.array(y,dtype='int')
        X = np.array(X)

        #train test set
        sss = StratifiedShuffleSplit(y, n_iter=10, test_size=0.25, random_state=19)
        for train_set_index, test_set_index in sss:
            X_train, y_train = X[train_set_index], y[train_set_index]
            X_test , y_test  = X[test_set_index] , y[test_set_index]
            break;

        #dimension reduction
        #LDA weights
        lda = LinearDiscriminantAnalysis(n_components=3)
        lda_weights = lda.fit(X_train, y_train)
        
        X_train = lda_weights.transform(X_train)
        X_test = lda_weights.transform(X_test)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

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