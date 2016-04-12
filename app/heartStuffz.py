import pickle

from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt



DATASET_LOCATION = "C:/dataset"
person = 1
Fs = 128 #global const frequency of the brain signals
nyq  = 0.5 * Fs

fname = str(DATASET_LOCATION) + '/s'
if person < 10:
    fname += '0'
fname += str(person) + '.dat'
with open(fname, 'rb') as f:
    p = pickle._Unpickler(f)
    p.encoding = ('latin1')
    data = p.load()

    # structure of data element:
    # data['labels'][video] = [valence, arousal, dominance, liking]
    # data['data'][video][channel] = [samples * 8064]
    channel = data['data'][17][38]
    channel = channel[:int(len(channel)/4)]

    # graphical output...
    plt.plot(np.arange(len(channel)) / Fs,channel, 'r-')
    plt.title("before")
    plt.xlabel("Time (s)")
    plt.ylabel("plethysmograph")
    plt.legend()
    plt.savefig('before')
    plt.show()
    plt.clf()
    plt.close()


    # lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
    low = 3 / nyq  # lower values => smoother
    b, a = butter(6, low, btype='low')
    channel = lfilter(b, a, channel)

    # Plethysmograph => heart rate, heart rate variability
    # this requires sufficient smoothing !!
    # heart rate is visible with local optima, therefore we need to search the optima first

    # stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    diffs = np.diff(np.sign(np.diff(channel)))
    extrema =  diffs.nonzero()[0] + 1 # local min+max
    minima  = (diffs > 0).nonzero()[0] + 1 # local min
    maxima  = (diffs < 0).nonzero()[0] + 1  # local max

    # graphical output...
    plt.plot(np.arange(len(channel))/Fs,channel, 'r-')
    plt.plot(minima/Fs, channel[minima], "o", label="min")
    plt.plot(maxima/Fs, channel[maxima], "o", label="max")
    plt.xlabel("Time (s)")
    plt.ylabel("plethysmograph")
    plt.title("extrema")
    plt.legend()
    plt.savefig('extrema')
    plt.show()
    plt.clf()
    plt.close()