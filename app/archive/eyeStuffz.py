import numpy as np
import pickle
import matplotlib.pyplot as plt


X, y = [], []

person = 1
print('loading person ' + str(person))
fname = str('C:/dataset') + '/s'
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
    eyeData = np.array(data['data'])
    eyeData = eyeData[0,33]

    #stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    diffs = np.diff(np.sign(np.diff(eyeData)))
    extrema =  diffs.nonzero()[0] + 1 # local min+max
    minima  = (diffs > 0).nonzero()[0] + 1 # local min
    maxima  = (diffs < 0).nonzero()[0] + 1 # local max

    # graphical output...
    plt.plot(np.arange(len(eyeData)),eyeData, 'r-')
    #plt.plot(minima, eyeData[minima], "o", label="min")
    #plt.plot(maxima, eyeData[maxima], "o", label="max")
    plt.legend()
    plt.show()
