from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import os
import pickle
from scipy.signal import butter, lfilter

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

def featureFunc(samples):
    #return np.array(samples[:][:32][:]) #only use EEG channels
    all_left_channels  = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3']
    all_right_channels = ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4']

    Fs = 128 #samples have freq 128Hz
    n = 8064 #number of samples
    nyq  = 0.5 * Fs
    low  = startFreq['alpha'] / nyq
    high = stopFreq['beta']   / nyq
    b, a = butter(6, [low, high], btype='band')
    ham = np.hamming(n)

    features = []
    for left, right in zip(all_left_channels, all_right_channels):
        #bandpass filter to get waveband        
        left_samples  = lfilter(b, a, samples[channelNames[left ]])    
        right_samples = lfilter(b, a, samples[channelNames[right]])

        #hamming window to smoothen edges    
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

        #features.append( [(left_avg - right_avg) / (right_avg + left_avg)] )
        features.append(left_avg )
        features.append(right_avg)

    return np.array(features)
def csp(samples, labels):
    #based on http://lib.ugent.be/fulltxt/RUG01/001/805/425/RUG01-001805425_2012_0001_AC.pdf    
    videoCount = len(labels)
    inputChannelCount = len(samples[0])
    sampleLength = len(samples[0][0]) #should be 8064

    #divide in two classes
    E = [np.zeros(( inputChannelCount,inputChannelCount ))] *2
    for klass, sample in zip(labels, samples):
        #each sample = 32 channels x 8064 samples
        E[klass] += np.cov(sample)

    #step one
    E_combined = E[0] + E[1]

    #step two Vt * E * V = P
    P, V = np.linalg.eig(E_combined)
    #put eigen values on diagonal => after **-.5 in step three
    
    #step three whitening
    U = np.dot( (P**-0.5)*np.eye(len(P)), np.transpose(V) )
    Rnul = U * E[0] * np.transpose(U)
    Reen = U * E[1] * np.transpose(U)

    #step four Zt * R0 * Z = D
    D, Z = np.linalg.eig(Rnul)
    #sorteer eigenwaarden op diagonaal
    idx = D.argsort()[::-1]   
    Z = Z[:,idx]
    D = D[idx]
    #put eigen values on diagonal
    D = D * np.eye(len(D))
    
    #step five
    W = np.transpose(Z) * U
    #print('Used filter coefs:', W)

    #apply filters
    features_CSP = []
    for sample in samples:
        features_CSP.append( np.dot(W, sample) )
    features_CSP = np.array(features_CSP)

    #apply csp
    #keep only 2 outer channels as these contain the most information
    X_only = np.zeros((videoCount,2,sampleLength))
    X_only[:,0,:] = features_CSP[:,0,:]
    X_only[:,1,:] = features_CSP[:,31,:]
    
    return np.array(X_only)

def loadPerson(person, preprocessFunc=csp, featureFunc=featureFunc, pad='../dataset'):
    X, y = [], []

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

        valences = np.array( data['labels'][:,0] ) #ATM only valence needed
        valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1

        #transform into classes
        #median
        median = np.median(valences)

        y = valences
        y[ y <= median ] = 0
        y[ y >  median ] = 1
        y = np.array(y, dtype='int')
        
        #preprocess data
        samples = np.array(data['data'])[:,:32,:] #throw out non-EEG channels
        X = preprocessFunc(samples=samples, labels=y)

        #extract features for each video
        for video in range(len(X)):
            X[video] = featureFunc(X[video])
        
    return [np.array(X), y]

if __name__ == "__main__":
    for person in range(1,33):
        #split using median, use all data
        X, y = loadPerson(person, featureFunc=featureFunc)
        #print(len(X), 'x', len(X[0]), 'x', len(X[0][0]))



        #LDA
        lda = LinearDiscriminantAnalysis()
        
        print(X_only.shape)
        print(y.shape)

        lda = lda.fit(X_only, y)
        print(lda)
        
        #accuracy
        #tptnfpfn
        #auc