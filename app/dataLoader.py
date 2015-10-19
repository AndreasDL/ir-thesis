import pickle
import numpy as np
from sklearn import linear_model
import featureExtractor as FE

from pprint import pprint

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
				features[j] = FE.calculateFeatures(data['data'][j])

	return [features, labels]