import numpy as np
from sklearn import preprocessing as PREP
from scipy.signal import butter, lfilter
#import matplotlib.pyplot as plt

#global const vars!
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
all_left_channels  = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3']
all_right_channels = ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4']

#turn bands into frequency ranges
startFreq = {'alpha' : 8 , 'beta'  : 13, 'gamma' : 30, 'delta' : 0, 'theta' : 4}
stopFreq  = {'alpha' : 13, 'beta'  : 30, 'gamma' : 50, 'delta' : 4, 'theta' : 8}

def powers(samples,waveband):
    if not (waveband in startFreq and waveband in stopFreq):
        print("Error Wrong waveband selection for frequencyPower. you selected:", waveband)    
        exit(-1)

    #it says left & right, but this has no meaning after CSP
    Fs = 128 #samples have freq 128Hz
    n = 8064 #number of samples
    features = []
    for channel in samples:
        #bandpass filter to get waveband
        nyq  = 0.5 * Fs 
        low  = startFreq[waveband] / nyq
        high = stopFreq[waveband]  / nyq
        b, a = butter(6, [low, high], btype='band')
        sample = lfilter(b, a, channel)

        #hamming window to smoothen edges
        ham = np.hamming(n)
        sample = np.multiply(sample , ham)

        #fft => get power components
        # fft computing and normalization
        Y = np.fft.fft(sample)/n
        Y = Y[range(round(n/2))]

        #average within one chunck
        avg = 0
        for val in Y:
            avg  += abs(val) **2

        avg /= len(Y)
        avg = np.sqrt(avg)

        features.append(avg)

    return np.array(features)

def alphaPowers(samples):
    return powers(samples,'alpha')
def betaPowers(samples):
    return powers(samples,'beta')
def gammaPowers(samples):
    return powers(samples,'gamma')
def deltaPowers(samples):
    return powers(samples,'delta')
def thetaPowers(samples):
    return powers(samples,'theta')

#old
def getBandPDChunks(waveband, samplesAtChannel, intervalLength=2, overlap=0.75 ):
    #splits the samlesAtChannel into chuncks of intervalLength size and calculates the frequency powers of a certain waveband using the fast fourier transform
    if not (waveband in startFreq and waveband in stopFreq):
        print("Error Wrong waveband selection for frequencyPower. you selected:", waveband)    
        exit(-1)

    if overlap > 1:
        print("Error: the overlap cannot be greater than 100 percent!")
        exit(-1)

    Fs = 128 #samples have freq 128Hz
    intervalsize = intervalLength * Fs #size of one chunk
    retArr = np.empty(0)

    #75% overlap => each chunck starts intervalsize/4 later
    for startIndex in range( 0, len(samplesAtChannel), round(intervalsize * (1-overlap)) ):

        stopIndex = startIndex + intervalsize
        samples = samplesAtChannel[startIndex:stopIndex]
        n = len(samples) #equal to intervalsize, except for last part
        
        #bandpass filter to get waveband
        nyq = 0.5 * Fs
        low = startFreq[waveband] / nyq
        high = stopFreq[waveband] / nyq
        b, a = butter(6, [low, high], btype='band')
        samples = lfilter(b, a, samples)    

        #hamming window to smoothen edges
        ham = np.hamming(n)
        samples = np.multiply(samples, ham)

        #fft => get power components
        # fft computing and normalization
        Y = np.fft.fft(samples)/n
        Y = Y[range(round(n/2))]

        #average within one chunck
        avg = 0
        for i in range(len(Y)):
            avg += abs(Y[i]) **2
        avg /= len(Y)
        avg = np.sqrt(avg)

        #add to values
        retArr = np.append(retArr, avg )

    #show the power spectrum
    '''
    frq = np.arange(n)*Fs/n # two sides frequency range
    freq = frq[range(round(n/2))]           # one side frequency range
    plt.plot(freq, abs(Y), 'r-')
    plt.xlabel('freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.show()

    exit()
    '''

    return retArr
def LMinRFraction(samples,intervalLength=2, overlap=0.75, 
    left_channels=['Fp1', 'AF3', 'F3', 'F7'],
    right_channels=['Fp2', 'AF4', 'F4', 'F8']
    ):
    
    #structure of samples[channel, sample]
    #return L-R / L+R, voor alpha components zie gegeven paper p6
    for left_channel, right_channel in zip(left_channels, right_channels):
        alpha_left  = getBandPDChunks('alpha', samples[channelNames[left_channel]], intervalLength, overlap )
        alpha_right = getBandPDChunks('alpha', samples[channelNames[right_channel]], intervalLength, overlap )

    return np.divide( alpha_left-alpha_right, alpha_left+alpha_right )

def LogLMinRAlpha(samples,intervalLength=2, overlap=0.75, left_channel='F3', right_channel='F4'):
    #log(left) - log(right)
    alpha_left  = getBandPDChunks('alpha', samples[channelNames[left_channel]], intervalLength, overlap )
    alpha_right = getBandPDChunks('alpha', samples[channelNames[right_channel]], intervalLength, overlap )

    #log left - log right
    return np.log(alpha_left) - np.log(alpha_right)
def FrontlineMidlineThetaPower(samples, channels, intervalLength=2, overlap=0.75):
    #frontal midline theta power is increase by positive emotion
    #structure of samples[channel, sample]
    power = getBandPDChunks('theta', samples[channelNames[channels[0]]], intervalLength, overlap)
    for channel in channels[1:]:
        power += getBandPDChunks('theta', samples[channelNames[channel]], intervalLength, overlap)

    return power