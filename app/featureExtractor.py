import numpy as np

channelNames = {
	'Fp1' : 1,
	'AF3' : 2,
	'F3'  : 3,
	'F7'  : 4,
	'FC5' : 5,
	'FC1' : 6,
	'C3'  : 7,
	'T7'  : 8,
	'CP5' : 9,
	'CP1' : 10,
	'P3'  : 11,
	'P7'  : 12,
	'PO3' : 13,
	'O1'  : 14,
	'Oz'  : 15,
	'Pz'  : 16,
	'Fp2' : 17,
	'AF4' : 18,
	'Fz'  : 19,
	'F4'  : 20,
	'F8'  : 21,
	'FC6' : 22,
	'FC2' : 23,
	'Cz'  : 24,
	'C4'  : 25,
	'T8'  : 26,
	'CP6' : 27,
	'CP2' : 28,
	'P4'  : 29,
	'P8'  : 30,
	'PO4' : 31,
	'O2'  : 32,
	'hEOG' : 33, #(horizontal EOG:  hEOG1 - hEOG2)	
	'vEOG' : 34, #(vertical EOG:  vEOG1 - vEOG2)
	'zEMG' : 35, #(Zygomaticus Major EMG:  zEMG1 - zEMG2)
	'tEMG' : 36, #(Trapezius EMG:  tEMG1 - tEMG2)
	'GSR'  : 37, #(values from Twente converted to Geneva format (Ohm))
	'Respiration belt' : 38,
	'Plethysmograph' : 39,
	'Temperature' : 40
}

def getFrequencyPower(waveband, samplesAtChannel,offsetStartTime = 0,offsetStopTime = 63):
	#turn bands into frequency ranges
	startFreq = {'alpha' : 8, 'beta'  : 13, 'gamma' : 30, 'delta' : 0, 'theta' : 4}
	stopFreq = {'alpha' : 13, 'beta'  : 30, 'gamma' : 50, 'delta' : 4, 'theta' : 8}

	#uses FFT to get the power of a certain waveband for a certain channel sample
	#E.G.: sum Alpha components for left and right
	#alpha runs from 8 - 13Hz
	#freq = index * Fs / n => value = abs(Y[j])
	#8hz = index * Fs/n <=> index = 8hz * 4032 / 128


	#FFT to get frequency components
	Fs = 128 #samples have freq 128Hz
	offsetStartIndex = offsetStartTime * Fs       #fix offset
	offestStopIndex = offsetStopTime * Fs
	samplesAtOffset = samplesAtChannel[offsetStartIndex:offestStopIndex]
	n = len(samplesAtOffset)

	if not (waveband in startFreq and waveband in stopFreq):
		print("Error Wrong waveband selection for frequencyPower")	
		exit(-1)

	startIndex  = round(startFreq[waveband]  * n / Fs)
	stopIndex   = round(stopFreq[waveband] * n / Fs)

	Y = np.fft.fft(samplesAtOffset)/n 					# fft computing and normalization
	Y = Y[range(round(n/2))]

	component = 0
	for i in range(startIndex,stopIndex+1):
		component += abs(Y[i])

	return component


def LRFraction(samples,offsetStartTime=0,offsetStopTime=63):
	#structure of samples[channel, sample]
	#return L-R / L+R, voor alpha components zie gegeven paper p6
	alpha_left = 0
	for i in [channelNames['F3'], channelNames['C3'], channelNames['P3']]:
		alpha_left += getFrequencyPower('alpha',samples[i],offsetStartTime,offsetStopTime)

	alpha_right = 0
	for i in [channelNames['F4'], channelNames['C4'], channelNames['P4']]:
		alpha_right += getFrequencyPower('alpha',samples[i],offsetStartTime,offsetStopTime)

	return [ (alpha_left - alpha_right) / (alpha_left + alpha_right) ]
	
def FrontlineMidlineThetaPower(samples,offsetStartTime=0,offsetStopTime=63):
	#frontal midline theta power is increase by positive emotion
	#structure of samples[channel, sample]
	
	power = 0
	for i in [channelNames['Fz'], channelNames['Cz'], channelNames['FC1'], channelNames['FC2']]:
		power += getFrequencyPower('theta', samples[i],offsetStartTime,offsetStopTime)

	return power

def leftMeanAlphaPower(samples,offsetStartTime=0,offsetStopTime=63):
	#structure of samples[channel, sample]
	#return L-R / L+R, voor alpha components zie gegeven paper p6
	alpha_left = 0
	for i in [channelNames['F3'], channelNames['C3'], channelNames['P3']]:
		alpha_left += getFrequencyPower('alpha',samples[i],offsetStartTime,offsetStopTime)

	return alpha_left / 3

def rightMeanAlphaPower(samples,offsetStartTime=0,offsetStopTime=63):
	#structure of samples[channel, sample]
	#return L-R / L+R, voor alpha components zie gegeven paper p6
	alpha_right = 0
	for i in [channelNames['F4'], channelNames['C4'], channelNames['P4']]:
		alpha_right += getFrequencyPower('alpha',samples[i],offsetStartTime,offsetStopTime)

	return alpha_right / 3

def alphaBetaRatio(samples,offsetStartTime=0,offsetStopTime=63):
	alpha = 0
	beta = 0
	for i in range(32):
		alpha += getFrequencyPower('alpha', samples[i])
		beta += getFrequencyPower('beta', samples[i])

	return alpha / beta



def calculateFeatures(samples):
	#return LRFraction(samples)
	#return FrontlineMidlineThetaPower(samples)
	#[leftMeanAlphaPower(samples,22,30), rightMeanAlphaPower(samples,22,30)]
	return alphaBetaRatio(samples)