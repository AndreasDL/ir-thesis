import numpy as np
from sklearn import preprocessing as PREP
from scipy.signal import butter, lfilter

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
startFreq = {'alpha' : 8 , 'beta'  : 13, 'gamma' : 30, 'delta' : 0, 'theta' : 4, 'all' : 0}
stopFreq  = {'alpha' : 13, 'beta'  : 30, 'gamma' : 50, 'delta' : 4, 'theta' : 8, 'all' : 50}

Fs = 128 #global const frequency of the brain signals
nyq  = 0.5 * Fs

class AFeatureExtractor:
    def __init__(self, featName):
        self.featureName = featName

    def extract(self, samples):
        '''This function should extract the features from the data
            Person & video specific; return features for each video
            so you get one pari of list<video> = [features]
        '''
        return []
    def getFeatureNames(self):
        '''This function should return the names of the features in the order that they occur
            Person specific; list<video> = [featurenames]
            this is need to find out which features were used
        '''
        return self.featureName

#EEG features
class PowerExtractor(AFeatureExtractor):
    '''this FE will axtract alpha and beta power from the brain and reports it ratio'''
    def __init__(self,,channels,freqBand, featName='Power'):
        AFeatureExtractor.__init__(self,featName)

        self.usedChannelIndexes = channels #list of channel indexes to use

        if not (freqBand in startFreq and freqBand in stopFreq):
            print("Error Wrong waveband selection for frequencyPower. you selected:", freqBand)
            exit(-1)
        self.usedFeqBand = freqBand

    def extract(self,samples):

        n = len(samples[0])#8064 #number of samples
        features = []
        for channel in self.usedChannelIndexes:
            #bandpass filter to get waveband
            low  = startFreq[self.usedFeqBand] / nyq
            high = stopFreq[ self.usedFeqBand] / nyq
            b, a = butter(6, [low, high], btype='band')
            sample = lfilter(b, a, channel)

            #hamming window to smoothen edges
            ham = np.hamming(n)
            sample = np.multiply(sample , ham)

            #fft => get power components
            # fft computing and normalization
            Y = np.fft.fft(sample)/n
            Y = Y[range(round(n/2))]

            #average
            avg = 0
            for val in Y:
                avg  += abs(val) **2

            avg /= len(Y)
            avg = np.sqrt(avg)

            features.append(avg)

        return np.array(features)
class AlphaBetaExtractor(AFeatureExtractor):
    def __init__(self,channels,featName='AlphaBeta'):
        AFeatureExtractor.__init__(self,featName)
        self.usedChannelIndexes = channels

    def extract(self, samples):
        alphaExtr = PowerExtractor(self.usedChannelIndexes, 'alpha')
        betaExtr  = PowerExtractor(self.usedChannelIndexes, 'beta')

        alpha_power = alphaExtr.extract(samples)
        beta_power  = betaExtr.extract(samples)

        return alpha_power / beta_power
class LMinRLPlusRExtractor(AFeatureExtractor):
    def __init__(self,left_channels, right_channels,featName='L-R/L+R'):
        AFeatureExtractor.__init__(self,featName)
        self.left_channels = left_channels
        self.right_channels = right_channels

        if len(left_channels) != len(right_channels):
            print('WARN left and right channels not of equal length')

    def extract(self,samples):
        leftExtr = PowerExtractor(self.left_channels,'alpha')
        rightExtr = PowerExtractor(self.right_channels, 'alpha')

        left_power = leftExtr.extract(samples)
        right_power = rightExtr.extract(samples)

        return (left_power - right_power) / (left_power + right_power)
class FrontalMidlinePower(PowerExtractor):
    def __init__(self,channels,featName='FM'):
        PowerExtractor.__init__(self,channels,'theta','FM')

#physiological features
class AvgExtractor(AFeatureExtractor):
    def __init__(self,channel,featName):
        if featName == ''
            featName = 'avg ' + channelNames(channel)

        AFeatureExtractor.__init__(self,featName)
        self.usedChannelIndex = channel

    def extract(self,samples):
        return np.average(samples[self.usedChannelIndex])
class STDExtractor(AFeatureExtractor):
    def __init__(self,channel,featName):
        if featName == ''
            featName = 'std ' + channelNames(channel)

        AFeatureExtractor.__init__(self,featName)
        self.usedChannelIndex = channel

    def extract(self,samples):
        return np.std(samples[self.usedChannelIndex])
#get Heart Rate from plethysmograph
class AVGHeartRateExtractor(AFeatureExtractor):
    def __init__(self,featName='avg HR'):
        AFeatureExtractor.__init__(self,featName)
        self.channel = channelNames['Plethysmograph']

    def extract(self,samples):
        #lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
        low  = 3 / nyq #lower values => smoother
        b, a = butter(6, low, btype='low')
        for channel in range(len(samples)):
            samples[channel] = lfilter(b, a, samples[channel])

        #Plethysmograph => heart rate, heart rate variability
        #this requires sufficient smoothing !!
        #heart rate is visible with local optima, therefore we need to search the optima first

        #stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        diffs = np.diff(np.sign(np.diff(samples[self.channel])))
        #extrema =  diffs.nonzero()[0] + 1 # local min+max
        #minima  = (diffs > 0).nonzero()[0] + 1 # local min
        maxima  = (diffs < 0).nonzero()[0] + 1 # local max

        # graphical output...
        #plt.plot(np.arange(len(data)),data, 'r-')
        #plt.plot(b, data[b], "o", label="min")
        #plt.plot(c, data[c], "o", label="max")
        #plt.legend()
        #plt.show()

        avg_rate = len(maxima)

        return avg_rate
class STDInterBeatExtractor(AFeatureExtractor):
    def __init__(self,featName='avg HR'):
        AFeatureExtractor.__init__(self,featName)
        self.channel = channelNames['Plethysmograph']

    def extract(self,samples):
        #lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
        low  = 3 / nyq #lower values => smoother
        b, a = butter(6, low, btype='low')
        for channel in range(len(samples)):
            samples[channel] = lfilter(b, a, samples[channel])

        #Plethysmograph => heart rate, heart rate variability
        #this requires sufficient smoothing !!
        #heart rate is visible with local optima, therefore we need to search the optima first

        #stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        diffs = np.diff(np.sign(np.diff(samples[self.channel])))
        #extrema =  diffs.nonzero()[0] + 1 # local min+max
        #minima  = (diffs > 0).nonzero()[0] + 1 # local min
        maxima  = (diffs < 0).nonzero()[0] + 1 # local max

        # graphical output...
        #plt.plot(np.arange(len(data)),data, 'r-')
        #plt.plot(b, data[b], "o", label="min")
        #plt.plot(c, data[c], "o", label="max")
        #plt.legend()
        #plt.show()


        interbeats = np.diff(maxima) / Fs #time in between beats => correlated to hreat rate!
        std_interbeats = np.var(interbeats) #std of beats => gives an estimations as to how much the heart rate varies


        return std_interbeats

class MultiFeatureExtractor(AFeatureExtractor):
    def __init__(self):
        self.featureExtrs = []

    def addFE(self,featureExtractor):
        self.featureExtrs.append(featureExtractor)

    def extract(self,samples):
        features = []

        for FE in self.featureExtrs:
            features.append(FE.extract(samples))

        return np.array(features)

    def getFeatureNames(self):
        names = []
        for FE in self.featureExtrs:
            names.append(FE.getFeatureNames())

        return names