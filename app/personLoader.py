import pickle
import random

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from archive import featureExtractor as FE
from archive.csp import Csp

class APersonLoader:
    def __init__(self, classFunc, featFunc, name, path='../dataset'):
        self.name = name
        self.path = path

        self.classFunc = classFunc
        self.featFunc  = featFunc

    def load(self,person):
        #return X_train, y_train, x_test, y_test
        return [], [], [], []

class PersonLoader(APersonLoader):
    def __init__(self, classFunc, featFunc, name='normal', path='../dataset' ):
        APersonLoader.__init__(self, name, classFunc, featFunc, path='../dataset')

    def load(self,person):
        fname = str(self.pad) + '/s'
        if person < 10:
            fname += '0'
        fname += str(person) + '.dat'
        with open(fname,'rb') as f:
            p = pickle._Unpickler(f)
            p.encoding= ('latin1')
            data = p.load()

            '''if plots:
                plotClass(data, person)
            '''

            #structure of data element:
            #data['labels'][video] = [valence, arousal, dominance, liking]
            #data['data'][video][channel] = [samples * 8064]

            X = self.featFunc(data)
            y = self.classFunc(data)

            #split train / test
            #n_iter = 1 => abuse the shuffle split, to obtain a static break, instead of crossvalidation
            sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=19)
            for train_set_index, test_set_index in sss:
                X_train, y_train = X[train_set_index], y[train_set_index]
                X_test , y_test  = X[test_set_index] , y[test_set_index]

            #fit normalizer to train set & normalize both train and testset
            #normer = Normalizer(copy=False)
            #normer.fit(X_train, y_train)
            #X_train = normer.transform(X_train, y_train, copy=False)
            #X_test  = normer.transform(X_test, copy=False)

            return X_train, y_train, X_test, y_test

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