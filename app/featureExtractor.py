import numpy as np
from sklearn import preprocessing as PREP
from scipy.signal import butter, lfilter
#import matplotlib.pyplot as plt

#global const vars!
channelNames = {
    'Fp1' : 1 , 'AF3' : 2 , 'F3'  : 3 , 'F7'  : 4 , 'FC5' : 5 , 'FC1' : 6 , 'C3'  : 7 , 'T7'  : 8 , 'CP5' : 9 , 'CP1' : 10, 
    'P3'  : 11, 'P7'  : 12, 'PO3' : 13, 'O1'  : 14, 'Oz'  : 15, 'Pz'  : 16, 'Fp2' : 17, 'AF4' : 18, 'Fz'  : 19, 'F4'  : 20,
    'F8'  : 21, 'FC6' : 22, 'FC2' : 23, 'Cz'  : 24, 'C4'  : 25, 'T8'  : 26, 'CP6' : 27, 'CP2' : 28, 'P4'  : 29, 'P8'  : 30,
    'PO4' : 31, 'O2'  : 32, 
    'hEOG' : 33, #(horizontal EOG:  hEOG1 - hEOG2)    
    'vEOG' : 34, #(vertical EOG:  vEOG1 - vEOG2)
    'zEMG' : 35, #(Zygomaticus Major EMG:  zEMG1 - zEMG2)
    'tEMG' : 36, #(Trapezius EMG:  tEMG1 - tEMG2)
    'GSR'  : 37, #(values from Twente converted to Geneva format (Ohm))
    'Respiration belt' : 38,
    'Plethysmograph' : 39,
    'Temperature' : 40
}
all_left_channels  = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3']
all_right_channels = ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4']

#turn bands into frequency ranges
startFreq = {'alpha' : 8, 'beta'  : 13, 'gamma' : 30, 'delta' : 0, 'theta' : 4}
stopFreq = {'alpha' : 13, 'beta'  : 30, 'gamma' : 50, 'delta' : 4, 'theta' : 8}

def powers(samples,waveband):
    if not (waveband in startFreq and waveband in stopFreq):
        print("Error Wrong waveband selection for frequencyPower. you selected:", waveband)    
        exit(-1)

    #it says left & right, but this has no meaning after CSP
    Fs = 128 #samples have freq 128Hz
    n = 8064 #number of samples

    #bandpass filter to get waveband
    nyq  = 0.5 * Fs 
    low  = startFreq[waveband] / nyq
    high = stopFreq[waveband]  / nyq
    b, a = butter(6, [low, high], btype='band')
    left_samples  = lfilter(b, a, samples[0])    
    right_samples = lfilter(b, a, samples[1])

    #hamming window to smoothen edges
    ham = np.hamming(n)
    left_samples  = np.multiply(left_samples , ham)
    right_samples = np.multiply(right_samples, ham)

    #fft => get power components
    # fft computing and normalization
    left_Y = np.fft.fft(left_samples)/n
    left_Y = left_Y[range(round(n/2))]
    right_Y = np.fft.fft(right_samples)/n
    right_Y = right_Y[range(round(n/2))]

    #average within one chunck
    left_avg, right_avg = 0, 0
    for left_val, right_val in zip(left_Y, right_Y):
        left_avg  += abs(left_val) **2
        right_avg += abs(right_val) **2

    left_avg /= len(left_Y)
    left_avg = np.sqrt(left_avg)
    right_avg /= len(right_Y)
    right_avg = np.sqrt(right_avg)

    features = []
    features.append(left_avg )
    features.append(right_avg)

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
def LMinRFraction(samples,intervalLength=2, overlap=0.75, left_channel='F3', right_channel='F4'):
    #structure of samples[channel, sample]
    #return L-R / L+R, voor alpha components zie gegeven paper p6
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