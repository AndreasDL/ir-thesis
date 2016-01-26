import os
import pickle
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import normalize
from sklearn.cross_validation import StratifiedShuffleSplit

import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 30}
matplotlib.rc('font', **font)


channelNames = {
#after removing EEG channels
    'GSR'  : 0, #(values from Twente converted to Geneva format (Ohm))
    'Respiration belt' : 1,
    'Plethysmograph' : 2,
    'Temperature' : 3
}
Fs = 128 #samples have freq 128Hz


def plot(data, descr="value"):
    n = len(data)
    plt.plot(np.arange(n)/128, data, 'r-')
    plt.xlabel('time (s)')
    plt.ylabel(descr)
    plt.show()


def heartStuffz(plethysmoData):
    #Plethysmograph => heart rate, inter beat times, heart rate variability
    #this requires sufficient smoothing !!

    #stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    diffs = np.diff(np.sign(np.diff(plethysmoData)))
    extrema =  diffs.nonzero()[0] + 1 # local min+max
    #minima  = (diffs > 0).nonzero()[0] + 1 # local min
    maxima  = (diffs < 0).nonzero()[0] + 1 # local max

    # graphical output...
    #plt.plot(np.arange(len(data)),data, 'r-')
    #plt.plot(b, data[b], "o", label="min")
    #plt.plot(c, data[c], "o", label="max")
    #plt.legend()
    #plt.show()

    avg_rate = len(extrema) / 2

    interbeats = np.diff(maxima) / Fs #time in between beats => correlated to hreat rate!
    std_interbeats = np.var(interbeats) #std of beats => gives an estimations as to how much the heart rate varies
    

    return avg_rate, std_interbeats


def classFunc(data):

    #labels
    valences = np.array( data['labels'][:,0] ) #ATM only valence needed
    #valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1
    valences[ valences <= 5 ] = 0
    valences[ valences >  5 ] = 1

    arousals = np.array( data['labels'][:,1] )
    #arousals = (arousals - 1) / 8 #1->9 to 0->8 to 0->1
    arousals[ arousals <= 5 ] = 0
    arousals[ arousals >  5 ] = 1

    #assign classes
    #              | low valence | high valence |
    #low  arrousal |      0      |       2      |
    #high arrousal |      1      |       3      |
    y = np.zeros(len(valences))
    for i, (val, arr) in enumerate(zip(valences, arousals)):
        y[i] = (val * 2) + arr

    return y
def featureFunc(data):
    #filter out right channels
    samples = np.array(data['data'])[:,36:,:] #throw out EEG channels & eye and muscle movement
    
    #lowpass filter
    Fs = 128 #samples have freq 128Hz
    n = len(samples[0])#8064 #number of samples
    nyq  = 0.5 * Fs 
    low  = 3 / nyq #lower values => smoother
    b, a = butter(6, low, btype='low')
    for video in range(len(samples)):
        for channel in range(len(samples[video])):
            samples[video][channel] = lfilter(b, a, samples[video][channel])

    #normalize
    for video in range(len(samples)):
        for channel in range(len(samples[video])):
            samples[video][channel] -= np.mean( samples[video][channel] )
            samples[video][channel] /= np.std(  samples[video][channel] )

    #features
    features = []
    for video in samples:
        video_features = []

        for channel in video:
            minVal = np.min( channel )
            maxVal = np.max( channel )

            video_features.append( np.mean(  channel) )
            video_features.append( np.std(   channel) )
            video_features.append( np.median(channel) )
            video_features.append( minVal )
            video_features.append( maxVal )
            video_features.append( maxVal - minVal )
        
        video_features.extend(
            heartStuffz(
                video[channelNames['Plethysmograph']]
            )
        )

        features.append(video_features)

    return np.array(features)

def loadPerson(person, classFunc, featureFunc, pad='../dataset'):
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

        X = featureFunc(data)
        y = classFunc(data)
        
        #split train / test 
        #n_iter = 1 => abuse the shuffle split, to obtain a static break, instead of crossvalidation
        sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=19)
        for train_set_index, test_set_index in sss:
            X_train, y_train = X[train_set_index], y[train_set_index]
            X_test , y_test  = X[test_set_index] , y[test_set_index]
        
        #anova => feature selection

        #model


if __name__ == '__main__':
    loadPerson(
        person = 1,
        classFunc = classFunc,
        featureFunc = featureFunc
    )