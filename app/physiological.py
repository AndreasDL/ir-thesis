import os
import pickle
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 30}
matplotlib.rc('font', **font)

#load data from disk
channelNames = {
#    'Fp1' : 0 , 'AF3' : 1 , 'F3'  : 2 , 'F7'  : 3 , 'FC5' : 4 , 'FC1' : 5 , 'C3'  : 6 , 'T7'  : 7 , 'CP5' : 8 , 'CP1' : 9, 
#    'P3'  : 10, 'P7'  : 11, 'PO3' : 12, 'O1'  : 13, 'Oz'  : 14, 'Pz'  : 15, 'Fp2' : 16, 'AF4' : 17, 'Fz'  : 18, 'F4'  : 19,
#    'F8'  : 20, 'FC6' : 21, 'FC2' : 22, 'Cz'  : 23, 'C4'  : 24, 'T8'  : 25, 'CP6' : 26, 'CP2' : 27, 'P4'  : 28, 'P8'  : 29,
#    'PO4' : 30, 'O2'  : 31, 
#    'hEOG' : 32, #(horizontal EOG:  hEOG1 - hEOG2)    
#    'vEOG' : 33, #(vertical EOG:  vEOG1 - vEOG2)
#    'zEMG' : 34, #(Zygomaticus Major EMG:  zEMG1 - zEMG2)
#    'tEMG' : 35, #(Trapezius EMG:  tEMG1 - tEMG2)
#    'GSR'  : 36, #(values from Twente converted to Geneva format (Ohm))
#    'Respiration belt' : 37,
#    'Plethysmograph' : 38,
#    'Temperature' : 39
#after removing EEG channels
    'GSR'  : 0, #(values from Twente converted to Geneva format (Ohm))
    'Respiration belt' : 1,
    'Plethysmograph' : 2,
    'Temperature' : 3
}

def loadPerson(person, pad='../dataset'):
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

        #filter out right channels
        samples = np.array(data['data'])[:,36:,:] #throw out EEG channels & eye and muscle movement
        
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
        #Plethysmograph => heart rate, inter beat times, heart rate variability
        #this requires sufficient smoothing
        localOpt(samples[0][2], genPlot=True)

        #other features
        #mean, variance/std, min, max, max - min, median, mode (value that occurs the most ?)

        #split train / test 

        #anova => feature selection

        #model

def plot(data, descr="value"):
    n = len(data)
    plt.plot(np.arange(n)/128, data, 'r-')
    plt.xlabel('time (s)')
    plt.ylabel(descr)
    plt.show()

def localOpt(data, genPlot=False):
    #stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    # that's the line, you need:
    diffs = np.diff(np.sign(np.diff(data)))
    a =  diffs.nonzero()[0] + 1 # local min+max
    b = (diffs > 0).nonzero()[0] + 1 # local min
    c = (diffs < 0).nonzero()[0] + 1 # local max

    if genPlot:
        # graphical output...
        plt.plot(np.arange(len(data)),data, 'r-')
        plt.plot(b, data[b], "o", label="min")
        plt.plot(c, data[c], "o", label="max")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    loadPerson(1)