import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

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
all_left_channel_names  = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3']
all_left_channels  = list(range(13))
all_right_channel_names = ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4']
all_right_channels = [16,17,19,20,21,22,24,25,26,27,28,29,30]

all_EEG_channels   = list(range(32))
all_phy_channels   = list(range(36,40))
all_FM_channels    = [18]

all_frontal_channel_names   = ['FC5', 'FC1','FC2', 'FC6', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Fp1','Fp2']
all_frontal_channels        = [channelNames[name] for name in all_frontal_channel_names]

all_posterior_channel_names = ['CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','O1','O2']
all_posterior_channels      = [channelNames[name] for name in all_posterior_channel_names]

all_channels = [#global const vars!
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
    'P3', 'P7' , 'PO3','O1', 'Oz', 'Pz', 'Fp2','AF4','Fz', 'F4',
    'F8', 'FC6','FC2','Cz' , 'C4', 'T8', 'CP6','CP2','P4', 'P8',
    'PO4','O2',
    'hEOG', #(horizontal EOG:  hEOG1 - hEOG2)
    'vEOG', #(vertical EOG:  vEOG1 - vEOG2)
    'zEMG', #(Zygomaticus Major EMG:  zEMG1 - zEMG2)
    'tEMG', #(Trapezius EMG:  tEMG1 - tEMG2)
    'GSR', #(values from Twente converted to Geneva format (Ohm))
    'Respiration belt',
    'Plethysmograph',
    'Temperature'
]

#turn bands into frequency ranges
startFreq = {'alpha' : 8 , 'beta'  : 13, 'gamma' : 30, 'delta' : 0, 'theta' : 4, 'all' : 0}
stopFreq  = {'alpha' : 13, 'beta'  : 30, 'gamma' : 50, 'delta' : 4, 'theta' : 8, 'all' : 50}

Fs = 128 #global const frequency of the brain signals
nyq  = 0.5 * Fs

class AFeatureExtractor:
    def __init__(self, featName):
        self.featureName = featName

    def extract(self, video):
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
class PSDExtractor(AFeatureExtractor):
    #Power spectral density of a channel
    def __init__(self,channels,freqBand, featName='Power'):
        AFeatureExtractor.__init__(self,featName)

        self.usedChannelIndexes = channels #list of channel indexes to use

        if not (freqBand in startFreq and freqBand in stopFreq):
            print("Error Wrong waveband selection for frequencyPower. you selected:", freqBand)
            exit(-1)
        self.usedFreqBand = freqBand

    def extract(self, video):

        n = len(video[0])#8064 #number of samples
        features = []
        for channelIndex in self.usedChannelIndexes:
            #get channel out the video data
            channel = video[channelIndex]

            #bandpass filter to get waveband
            low  = startFreq[self.usedFreqBand] / nyq
            high = stopFreq[ self.usedFreqBand] / nyq
            b, a = butter(6, [low, high], btype='band')
            sample = lfilter(b, a, channel)

            #hamming window to smoothen edges
            #ham = np.hamming(n)
            #sample = np.multiply(sample , ham)

            #fft => get power components
            # fft computing and normalization
            Y = np.fft.fft(sample)/n
            Y = Y[:round(n/2)]

            #average
            avg = 0
            for val in Y:
                avg += abs(val)**2

            avg /= float(len(Y))
            avg = np.sqrt(avg)

            features.append(avg)

        return np.array(features)
class DEExtractor(AFeatureExtractor):
    #Power spectral density of a channel
    def __init__(self,channels,freqBand, featName='Power'):
        AFeatureExtractor.__init__(self,featName)
        self.usedChannelIndexes = channels #list of channel indexes to use

        if not (freqBand in startFreq and freqBand in stopFreq):
            print("Error Wrong waveband selection for frequencyPower. you selected:", freqBand)
            exit(-1)
        self.usedFreqBand = freqBand



    def extract(self, video):
        n = len(video[0])#8064 #number of samples
        features = []
        for channelIndex in self.usedChannelIndexes:
            #get channel out the video data
            channel = video[channelIndex]

            #bandpass filter to get waveband
            low  = startFreq[self.usedFreqBand] / nyq
            high = stopFreq[ self.usedFreqBand] / nyq
            b, a = butter(6, [low, high], btype='band')
            sample = lfilter(b, a, channel)

            #average Energy spectrum
            avg = 0
            for val in sample:
                avg += abs(val)**2

            avg /= float(len(sample))
            avg = np.sqrt(avg)

            features.append(avg)

        return np.log(features)

class DASMExtractor(AFeatureExtractor):
    #DE left - DE right
    def __init__(self,left_channels, right_channels,freqBand,featName=''):
        AFeatureExtractor.__init__(self,featName)

        self.leftDEExtractor = DEExtractor(left_channels,freqBand,featName)
        self.rightDEExtractor = DEExtractor(right_channels,freqBand,featName)

        self.usedChannelIndexes= left_channels[:]
        self.usedChannelIndexes.extend(right_channels[:])

        self.usedFreqBand = freqBand

    def extract(self,video):
        leftDE  = self.leftDEExtractor.extract(video)
        rightDE = self.rightDEExtractor.extract(video)

        return leftDE - rightDE
class RASMExtractor(AFeatureExtractor):
    #DE left / DE right
    def __init__(self,left_channels, right_channels,freqBand,featName=''):
        AFeatureExtractor.__init__(self,featName)

        self.leftDEExtractor = DEExtractor(left_channels,freqBand,featName)
        self.rightDEExtractor = DEExtractor(right_channels,freqBand,featName)

        self.usedChannelIndexes = left_channels[:]
        self.usedChannelIndexes.extend(right_channels[:])

        self.usedFreqBand = freqBand

    def extract(self,video):
        leftDE  = self.leftDEExtractor.extract(video)
        rightDE = self.rightDEExtractor.extract(video)

        return float(leftDE) / float(rightDE)

class DCAUExtractor(AFeatureExtractor):
    #DE front -  DE posterior
    def __init__(self,frontal_channels, posterior_channels,freqBand,featName='AlphaBeta'):
        AFeatureExtractor.__init__(self,featName)

        self.frontalDEExtractor = DEExtractor(frontal_channels,freqBand,featName)
        self.posteriorDEExtractor = DEExtractor(posterior_channels,freqBand,featName)

        self.usedChannelIndexes = frontal_channels[:]
        self.usedChannelIndexes.extend(posterior_channels[:])
        self.usedFreqBand = freqBand
    def extract(self,video):
        frontalDE  = self.frontalDEExtractor.extract(video)
        posteriorDE = self.posteriorDEExtractor.extract(video)

        return frontalDE - posteriorDE
class RCAUExtractor(AFeatureExtractor):
    #DE front -  DE posterior
    def __init__(self,frontal_channels, posterior_channels,freqBand,featName='AlphaBeta'):
        AFeatureExtractor.__init__(self,featName)

        self.frontalDEExtractor = DEExtractor(frontal_channels,freqBand,featName)
        self.posteriorDEExtractor = DEExtractor(posterior_channels,freqBand,featName)
        self.usedFreqBand = freqBand

        self.usedChannelIndexes = frontal_channels[:]
        self.usedChannelIndexes.extend(posterior_channels[:])


def extract(self,video):
        frontalDE  = self.frontalDEExtractor.extract(video)
        posteriorDE = self.posteriorDEExtractor.extract(video)

        return float(frontalDE) / float(posteriorDE)

#more traditional
class AlphaBetaExtractor(AFeatureExtractor):
    def __init__(self,channels,featName='AlphaBeta'):
        AFeatureExtractor.__init__(self,featName)
        self.usedChannelIndexes = channels

    def extract(self, video):
        alphaExtr = PSDExtractor(self.usedChannelIndexes, 'alpha')
        betaExtr  = PSDExtractor(self.usedChannelIndexes, 'beta')

        alpha_power = np.sum(alphaExtr.extract(video))
        beta_power  = np.sum(betaExtr.extract(video))

        return alpha_power / float(beta_power)
class LMinRLPlusRExtractor(AFeatureExtractor):
    def __init__(self,left_channels, right_channels,featName='L-R/L+R'):
        AFeatureExtractor.__init__(self,featName)
        self.left_channels = left_channels
        self.right_channels = right_channels

        if len(left_channels) != len(right_channels):
            print('WARN left and right channels not of equal length')

    def extract(self, video):
        leftExtr  = PSDExtractor(self.left_channels, 'alpha')
        rightExtr = PSDExtractor(self.right_channels, 'alpha')

        left_power  = np.sum(leftExtr.extract(video))
        right_power = np.sum(rightExtr.extract(video))

        return (left_power - right_power) / float(left_power + right_power)
class FrontalMidlinePower(PSDExtractor):
    def __init__(self,channels,featName='FM'):
        PSDExtractor.__init__(self, channels, 'theta', featName)
#fractions
class BandFracExtractor(AFeatureExtractor):
    def __init__(self,channels,freqBand,featName):
        AFeatureExtractor.__init__(self,featName)
        self.usedChannelIndexes = channels
        self.usedFreqBand = freqBand

    def extract(self, video):
        all_power = float(np.sum(PSDExtractor(self.usedChannelIndexes, 'all').extract(video)))
        bnd_power = float(np.sum(PSDExtractor(self.usedChannelIndexes, self.usedFreqBand).extract(video)))

        return bnd_power / all_power


#physiological features
class AvgExtractor(AFeatureExtractor):
    def __init__(self,channel,featName):
        if featName == '':
            featName = 'avg ' + all_channels[channel]

        AFeatureExtractor.__init__(self,featName)
        self.usedChannelIndex = channel

    def extract(self, video):
        return np.average(video[self.usedChannelIndex])
class STDExtractor(AFeatureExtractor):
    def __init__(self,channel,featName):
        if featName == '':
            featName = 'std ' + all_channels[channel]

        AFeatureExtractor.__init__(self,featName)
        self.usedChannelIndex = channel

    def extract(self, video):
        return np.std(video[self.usedChannelIndex])
class MaxExtractor(AFeatureExtractor):
    def __init__(self, channel, featName):
        if featName == '':
            featName = 'max ' + all_channels[channel]

        AFeatureExtractor.__init__(self, featName)
        self.usedChannelIndex = channel

    def extract(self, video):
        return np.max(video[self.usedChannelIndex])
class MinExtractor(AFeatureExtractor):
    def __init__(self, channel, featName):
        if featName == '':
            featName = 'min ' + all_channels[channel]

        AFeatureExtractor.__init__(self, featName)
        self.usedChannelIndex = channel

    def extract(self, video):
        return np.min(video[self.usedChannelIndex])
class MedianExtractor(AFeatureExtractor):
    def __init__(self, channel, featName):
        if featName == '':
            featName = 'median ' + all_channels[channel]

        AFeatureExtractor.__init__(self, featName)
        self.usedChannelIndex = channel

    def extract(self, video):
        return np.median(video[self.usedChannelIndex])
class VarExtractor(AFeatureExtractor):
    def __init__(self, channel, featName):
        if featName == '':
            featName = 'var ' + all_channels[channel]

        AFeatureExtractor.__init__(self, featName)
        self.usedChannelIndex = channel

    def extract(self, video):
        return np.var(video[self.usedChannelIndex])


#get Heart Rate from plethysmograph
class AVGHeartRateExtractor(AFeatureExtractor):
    def __init__(self,featName='avg HR'):
        AFeatureExtractor.__init__(self,featName)
        self.channel = channelNames['Plethysmograph']

    def extract(self, video):
        #lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
        low  = 3 / nyq #lower values => smoother
        b, a = butter(6, low, btype='low')
        for channel in range(len(video)):
            video[channel] = lfilter(b, a, video[channel])

        #Plethysmograph => heart rate, heart rate variability
        #this requires sufficient smoothing !!
        #heart rate is visible with local optima, therefore we need to search the optima first

        #stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        diffs = np.diff(np.sign(np.diff(video[self.channel])))
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
    def __init__(self,featName='std HR'):
        AFeatureExtractor.__init__(self,featName)
        self.channel = channelNames['Plethysmograph']

    def extract(self, video):
        #lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
        low  = 3 / nyq #lower values => smoother
        b, a = butter(6, low, btype='low')
        for channel in range(len(video)):
            video[channel] = lfilter(b, a, video[channel])

        #Plethysmograph => heart rate, heart rate variability
        #this requires sufficient smoothing !!
        #heart rate is visible with local optima, therefore we need to search the optima first

        #stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        diffs = np.diff(np.sign(np.diff(video[self.channel])))
        #extrema =  diffs.nonzero()[0] + 1 # local min+max
        #minima  = (diffs > 0).nonzero()[0] + 1 # local min
        maxima  = (diffs < 0).nonzero()[0] + 1 # local max

        # graphical output...
        #plt.plot(np.arange(len(data)),data, 'r-')
        #plt.plot(b, data[b], "o", label="min")
        #plt.plot(c, data[c], "o", label="max")
        #plt.legend()
        #plt.show()


        interbeats = np.diff(maxima) / float(Fs) #time in between beats => correlated to hreat rate!
        std_interbeats = np.var(interbeats) #std of beats => gives an estimations as to how much the heart rate varies


        return std_interbeats
class MaxHRExtractor(AFeatureExtractor):
    def __init__(self, featName=''):
        if featName == '':
            featName = 'maxHR'

        AFeatureExtractor.__init__(self, featName)
        self.channel = channelNames['Plethysmograph']

    def extract(self, video):
        # lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
        low = 3 / nyq  # lower values => smoother
        b, a = butter(6, low, btype='low')
        for channel in range(len(video)):
            video[channel] = lfilter(b, a, video[channel])

        # Plethysmograph => heart rate, heart rate variability
        # this requires sufficient smoothing !!
        # heart rate is visible with local optima, therefore we need to search the optima first

        # stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        diffs = np.diff(np.sign(np.diff(video[self.channel])))
        maxima = (diffs < 0).nonzero()[0] + 1  # local max

        interbeats = np.diff(maxima) / float(Fs)  # time in between beats => correlated to hreat rate!

        return np.max(interbeats)
class MinHRExtractor(AFeatureExtractor):
    def __init__(self, featName=''):
        if featName == '':
            featName = 'minHR'

        AFeatureExtractor.__init__(self, featName)
        self.channel = channelNames['Plethysmograph']

    def extract(self, video):
        # lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
        low = 3 / nyq  # lower values => smoother
        b, a = butter(6, low, btype='low')
        for channel in range(len(video)):
            video[channel] = lfilter(b, a, video[channel])

        # Plethysmograph => heart rate, heart rate variability
        # this requires sufficient smoothing !!
        # heart rate is visible with local optima, therefore we need to search the optima first

        # stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        diffs = np.diff(np.sign(np.diff(video[self.channel])))
        maxima = (diffs < 0).nonzero()[0] + 1  # local max

        interbeats = np.diff(maxima) / float(Fs)  # time in between beats => correlated to hreat rate!

        return np.min(interbeats)
class MedianHRExtractor(AFeatureExtractor):
    def __init__(self, featName=''):
        if featName == '':
            featName = 'medianHR'

        AFeatureExtractor.__init__(self, featName)
        self.channel = channelNames['Plethysmograph']

    def extract(self, video):
        # lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
        low = 3 / nyq  # lower values => smoother
        b, a = butter(6, low, btype='low')
        for channel in range(len(video)):
            video[channel] = lfilter(b, a, video[channel])

        # Plethysmograph => heart rate, heart rate variability
        # this requires sufficient smoothing !!
        # heart rate is visible with local optima, therefore we need to search the optima first

        # stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        diffs = np.diff(np.sign(np.diff(video[self.channel])))
        maxima = (diffs < 0).nonzero()[0] + 1  # local max

        interbeats = np.diff(maxima) / float(Fs)  # time in between beats => correlated to hreat rate!

        return np.median(interbeats)
class VarHRExtractor(AFeatureExtractor):
    def __init__(self, featName=''):
        if featName == '':
            featName = 'varHR'

        AFeatureExtractor.__init__(self, featName)
        self.channel = channelNames['Plethysmograph']

    def extract(self, video):
        # lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
        low = 3 / nyq  # lower values => smoother
        b, a = butter(6, low, btype='low')
        for channel in range(len(video)):
            video[channel] = lfilter(b, a, video[channel])

        # Plethysmograph => heart rate, heart rate variability
        # this requires sufficient smoothing !!
        # heart rate is visible with local optima, therefore we need to search the optima first

        # stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        diffs = np.diff(np.sign(np.diff(video[self.channel])))
        maxima = (diffs < 0).nonzero()[0] + 1  # local max

        interbeats = np.diff(maxima) / float(Fs)  # time in between beats => correlated to hreat rate!

        return np.var(interbeats)

#!! only one that can handle a list of videos!
class MultiFeatureExtractor(AFeatureExtractor):
    def __init__(self):
        self.featureExtrs = []

    def addFE(self,featureExtractor):
        self.featureExtrs.append(featureExtractor)

    def extract(self, video):
        retFeat = []
        for single_video in video:
            features = []
            for FE in self.featureExtrs:
                features.append(FE.extract(single_video))

            retFeat.append(features)

        return np.array(retFeat)

    def getFeatureNames(self):
        names = []
        for FE in self.featureExtrs:
            names.append(FE.getFeatureNames())

        return names