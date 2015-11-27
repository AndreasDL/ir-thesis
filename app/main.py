import os
import pickle
import util as UT
import numpy as np
import datetime
import time
from scipy.signal import butter, lfilter
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize

#global const vars!
'''
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
'''

def featureFunc(samples):
    #it says left & right, but this has no meaning after CSP
    
    Fs = 128 #samples have freq 128Hz
    n = 8064 #number of samples
    #turn bands into frequency ranges
    startFreq = {'alpha' : 8, 'beta'  : 13, 'gamma' : 30, 'delta' : 0, 'theta' : 4}
    stopFreq = {'alpha' : 13, 'beta'  : 30, 'gamma' : 50, 'delta' : 4, 'theta' : 8}

    #bandpass filter to get waveband
    nyq  = 0.5 * Fs
    low  = startFreq['alpha'] / nyq
    high = stopFreq['alpha']   / nyq
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
    #features.append( (left_avg - right_avg) / float(left_avg + right_avg) )

    return np.array(features)

#http://lib.ugent.be/fulltxt/RUG01/001/805/425/RUG01-001805425_2012_0001_AC.pdf
def own_csp(samples, labels):
    #sampes[video][channel] = list<samples>
    
    #normalize input => avoid problems with **-.5 of P matrix
    for i in range(32):
        samples[:,i,:] = normalize(samples[:,i,:])

    #divide in two classes
    cov0 = 0
    cov1 = 0
    for klass, sample in zip(labels, samples):
        #each sample = 32 channels x 8064 samples
        #avg = nul?
        E = sample
        Et = np.transpose(sample)
        
        prod = np.dot(E, Et)

        if klass == 0:
            cov0 += prod/np.trace(prod)
        else:
            cov1 += prod/np.trace(prod)
        
    #step one
    cov = cov0 + cov1

    #step two Vt * E * V = P
    P, V = np.linalg.eig(cov)
    P = np.diag(P)
    
    #step three whitening
    U = np.dot( np.sqrt(np.linalg.inv(P)), np.transpose(V) )
    Rnul = np.dot( U, np.dot(cov0, np.transpose(U)) )
    #Reen = np.dot( U, np.dot(cov1, np.transpose(U)) )

    #step four Zt * R0 * Z = D
    D, Z = np.linalg.eig(Rnul)
    
    #sorteer eigenwaarden op diagonaal
    idx = D.ravel().argsort()
    Z = Z[:,idx]
    D = D[idx]
    #put eigen values on diagonal
    D = np.diag(D)
    
    #step five
    W = np.dot( np.transpose(Z), U)
    #print(D)
    #print(W)

    #apply filters
    #TODO speedup only calculate 2 outer stuffz
    X_only = np.zeros((40,2,8064))
    for i in range(len(samples)):
        E = samples[i]
        #apply csp
        E_csp = np.dot(W, E)

        #keep only 2 outer channels as these contain the most information
        X_only[i,0,:] = E_csp[0,:]
        X_only[i,1,:] = E_csp[31,:]
    
    return X_only

def loadPerson(person, preprocessFunc=own_csp, featureFunc=featureFunc, pad='../dataset'):
    print('loadPerson')

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

        #only use EEG channels
        samples = np.array(data['data'])[:,:32,:] #throw out non-EEG channels

        #rescale
        valences = np.array( data['labels'][:,0] ) #ATM only valence needed
        valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1
        
        #median
        median = np.median(valences)
        #classes
        y = valences
        y[ y <= median ] = 0
        y[ y >  median ] = 1
        y = np.array(y, dtype='int')

        #preprocessing
        X_prepped = preprocessFunc(samples=samples,labels=y)

        #extract features for each video
        for video in range(len(data['data'])):
            X.append( featureFunc(X_prepped[video]) )

    return [np.array(X), y]

if __name__ == "__main__":
    CVSets = float(4)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    f = open("output" + str(st) + ".txt", 'w')
    f.write('CVSets: ' + str(CVSets) + '\n' +
        'person;acc;tp;tn;fp;fn;auc\n'
    )

    for person in range(1,33):
        #split using median, use all data
        X, y = loadPerson(person, 
            featureFunc=featureFunc,
            preprocessFunc=own_csp
        )

        #LDA
        lda = LinearDiscriminantAnalysis()
        K_CV = KFold(len(X), n_folds=CVSets, random_state=17, shuffle=True)
        acc, tp, tn , fp , fn , auc = 0, 0, 0, 0, 0, 0
        for train_index, CV_index in K_CV:
            lda = lda.fit(X[train_index], y[train_index])

            predictions = lda.predict(X[CV_index])
            #MSE train err
            (ttp,ttn,tfp,tfn) = UT.tptnfpfn(predictions, y[CV_index])
            tp  += ttp
            tn  += ttn
            fp  += tfp
            fn  += tfn

            acc += UT.accuracy(predictions, y[CV_index])
            auc += UT.auc(predictions, y[CV_index])

        #accuracy
        acc /= CVSets
        
        #tptnfpfn
        #results for 4 x 10 videos in CV!
        tp /= 40
        tn /= 40
        fp /= 40
        fn /= 40

        #auc
        auc /= CVSets

        print('person: ', person, 
            ' - acc: ', str(acc),
            ' - tp: ' , str(tp),
            ' - tn: ' , str(tn),
#            ' - fp: ' , str(fp),
#            ' - fn: ' , str(fn),
            ' - auc: ', str(auc)
        )
        f.write(str(person) + ';' + str(acc) + ';' + 
            str(tp) + ';' + str(tn) + ';' +
            str(fp) + ';' + str(fn) + ';' +
            str(auc) + '\n'
        )

    f.close()