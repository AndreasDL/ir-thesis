import os
import pickle
import numpy as np
from scipy.signal import butter, lfilter

#init
person = 7
startFreq = [1, 4, 8, 13, 36]
stopFreq = [4, 8, 13, 30, 40]
Fs = 128
nyq = 0.5 * Fs

#read data
fname = str('../dataset/s')
if person < 10:
    fname += '0'
fname += str(person) + '.dat'

with open(fname,'rb') as f:
    p = pickle._Unpickler(f)
    p.encoding= ('latin1')
    data = p.load()

    y = [[None] * 4 ] * len(data['data'])
    
    for videoIndex in range(len(data['data'])):
        video = data['data'][videoIndex]
        y[videoIndex] = data['labels'][videoIndex]

        features = [[list()] * 32] * 5
        #features[band][channel] = list<log band power for each sec>

        for channelIndex in range(32):
            channel = video[channelIndex]
            
            for startIndex in range(0, len(channel), Fs): #one feature per second
                samples = channel[startIndex:startIndex + Fs]
                n = len(samples)

                #get 5 power components
                power = [None] * 5
                for band in range(len(startFreq)):
                    #bandpass filter to get waveband
                    low = startFreq[band] / nyq
                    high = stopFreq[band] / nyq
                    b, a = butter(6, [low, high], btype='band')
                    sampl = lfilter(b, a, samples)

                    #fft => get power components
                    # fft computing and normalization
                    Y = np.fft.fft(sampl)/n
                    Y = Y[range(round(n/2))]

                    #log band energy?
                    power[band] = 0
                    for i in range(len(Y)):
                        power[band] += abs(Y[i])
                    features[band][channelIndex].append( np.log(power[band]) )
        
    print(y)
    print(features)
