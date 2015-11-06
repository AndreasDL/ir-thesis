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
#turn bands into frequency ranges
startFreq = {'alpha' : 8, 'beta'  : 13, 'gamma' : 30, 'delta' : 0, 'theta' : 4}
stopFreq = {'alpha' : 13, 'beta'  : 30, 'gamma' : 50, 'delta' : 4, 'theta' : 8}

#all features to be used
#samples =all channels of a single video
def extract(samples):
	return LMinRFraction(samples,2)

#Power Density of one band
def getBandPD(waveband, samplesAtChannel,offsetStartTime = 0,offsetStopTime = 63):

	if not (waveband in startFreq and waveband in stopFreq):
		print("Error Wrong waveband selection for frequencyPower. you selected:", waveband)	
		exit(-1)

	Fs = 128 #samples have freq 128Hz

	#hamming window to smoothen edges
	ham = np.hamming(len(samplesAtChannel))
	samplesAtChannel = np.multiply(samplesAtChannel, ham)


	#select only certain time
	offsetStartIndex = offsetStartTime * Fs       #fix offset
	offestStopIndex = offsetStopTime * Fs
	samplesAtOffset = samplesAtChannel[offsetStartIndex:offestStopIndex]
	n = len(samplesAtOffset)

	#bandpass filter to get waveband
	nyq = 0.5 * Fs
	low = startFreq[waveband] / nyq
	high = stopFreq[waveband] / nyq
	b, a = butter(6, [low, high], btype='band')
	samplesAtBand = lfilter(b, a, samplesAtOffset)	

	#fft => get components
	Y = np.fft.fft(samplesAtBand)/n					# fft computing and normalization
	Y = Y[range(round(n/2))]

	#show result of bandpass filter
	'''frq = np.arange(n)*Fs/n # two sides frequency range
	freq = frq[range(round(n/2))]           # one side frequency range
	plt.plot(freq, abs(Y), 'r-')
	plt.xlabel('freq (Hz)')
	plt.ylabel('|Y(freq)|')
	plt.show()'''

	#Root Mean Square
	value = 0
	for i in range(len(Y)):
		value += abs(Y[i]) **2

	return np.sqrt( value / len(Y) )
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

#valence
def LMinRFraction(samples,intervalLength=2, overlap=0.75):
	#structure of samples[channel, sample]
	#return L-R / L+R, voor alpha components zie gegeven paper p6
	alpha_left  = getBandPDChunks('alpha', samples[channelNames['F3']], intervalLength, overlap )
	alpha_right = getBandPDChunks('alpha', samples[channelNames['F4']], intervalLength, overlap )

	return np.divide( alpha_left-alpha_right, alpha_left+alpha_right )
	
def LogLMinRAlpha(samples, intervalLength=2,overlap=0.75):
	#log(left) - log(right)
	left_values  = getBandPDChucks('alpha', samples[channelNames['P3']], intervalLength, overlap)
	right_values = getBandPDChunks('alpha', samples[channelNames['P4']], intervalLength, overlap)

	#log left - log right
	return np.log(left_values) - np.log(right_values)

#arousal
def FrontlineMidlineThetaPower(samples,offsetStartTime=0,offsetStopTime=63):
	#frontal midline theta power is increase by positive emotion
	#structure of samples[channel, sample]
	
	power = 0
	for i in [channelNames['Fz'], channelNames['Cz'], channelNames['FC1'], channelNames['FC2']]:
		power += getBandPD('theta', samples[i],offsetStartTime,offsetStopTime)

	return power