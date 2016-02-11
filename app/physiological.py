import os
import pickle
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import util as UT
import featureExtractor as FE

from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 30}
matplotlib.rc('font', **font)


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

Fs = 128 #samples have freq 128Hz
nyq  = 0.5 * Fs


def heartStuffz(plethysmoData):
    #Plethysmograph => heart rate, heart rate variability
    #this requires sufficient smoothing !!
    #heart rate is visible with local optima, therefore we need to search the optima first

    #stolen with pride from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    diffs = np.diff(np.sign(np.diff(plethysmoData)))
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

    interbeats = np.diff(maxima) / Fs #time in between beats => correlated to hreat rate!
    std_interbeats = np.var(interbeats) #std of beats => gives an estimations as to how much the heart rate varies
    

    return avg_rate, std_interbeats
def classFunc(data):

    #labels
    valences = np.array( data['labels'][:,0] ) #ATM only valence needed
    #valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1
    valences[ valences <= 5 ] = 0
    valences[ valences >  5 ] = 1

    arousals = np.array( data['labels'][:,1] )
    #arousals = (arousals - 1) / 8 #1->9 to 0->8 to 0->1
    arousals[ arousals <= 5 ] = 0
    arousals[ arousals >  5 ] = 1

    #assign classes
    #              | low valence | high valence |
    #low  arrousal |      0      |       2      |
    #high arrousal |      1      |       3      |
    y = np.zeros(len(valences))
    for i, (val, arr) in enumerate(zip(valences, arousals)):
        y[i] = arr #(val * 2) + arr

    return y

def physiologicalFeatures(video):

    #filter out right channels
    samples = np.array(video)[36:,:] #throw out EEG channels & eye and muscle movement

    #physiological features
    #lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
    low  = 3 / nyq #lower values => smoother
    b, a = butter(6, low, btype='low')
    for channel in range(len(samples)):
        samples[channel] = lfilter(b, a, samples[channel])

    #extract features
    features = []
    for channel in samples:

        features.append( np.mean(  channel) )
        features.append( np.std(   channel) )
        #video_features.append( np.median(channel) )
        #video_features.append( minVal )
        #video_features.append( maxVal )
        #video_features.append( maxVal - minVal )

    features.extend(
        heartStuffz(
            video[channelNames['Plethysmograph']]
        )
    )

    #features look like this (person specific)
    #list = | avg_GSR | std_GSR | avg_rb  | std_rb  |
    #       | avg_ply | std_ply | avg_tem | std_tem |
    #       | avg_hr  | var_interbeats |

    return features

def eegFeatures(video):
    #EEG features
    features = []

    #alpha beta
    alpha_total = np.sum(FE.powers(video, 'alpha'))
    beta_total  = np.sum(FE.powers(video, 'beta' ))
    features.append(alpha_total / beta_total)

    #l-R / L+R
    left_alpha = 0
    right_alpha = 0
    for ch_left, ch_right in zip(all_left_channels, all_right_channels):
        left = []
        left.append( video[channelNames[ch_left],:] )
        left_alpha  += FE.powers(np.array(left) , 'alpha')[0]

        right = []
        right.append(video[channelNames[ch_right],:])
        right_alpha += FE.powers(np.array(right), 'alpha')[0]

    features.append( (left_alpha - right_alpha)/(left_alpha + right_alpha) )

    #FM
    frontal = []
    frontal.append(video[channelNames['Fz']])
    frontalMidlinePower = FE.powers( np.array(frontal),'theta')[0]
    features.append(frontalMidlinePower)

    #features look like this (person specific)
    #list = | alpha/beta | L-R/L+R | FM |

    return features

def featureFunc(data):
    samples = np.array(data['data'])

    features = []
    for video in samples:
        video_features = physiologicalFeatures(video)
        video_features.extend( eegFeatures(video) )

        features.append(video_features)

    return np.array(features)

def loadPerson(person, classFunc, featureFunc, pad='../dataset'):
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

        X = featureFunc(data)
        y = classFunc(data)
        
        #split train / test 
        #n_iter = 1 => abuse the shuffle split, to obtain a static break, instead of crossvalidation
        sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=19)
        for train_set_index, test_set_index in sss:
            X_train, y_train = X[train_set_index], y[train_set_index]
            X_test , y_test  = X[test_set_index] , y[test_set_index]
        
        #fit normalizer to train set & normalize both train and testset
        #normer = Normalizer(copy=False)
        #normer.fit(X_train, y_train)
        #X_train = normer.transform(X_train, y_train, copy=False)
        #X_test  = normer.transform(X_test, copy=False)

        return X_train, y_train, X_test, y_test

def PersonWorker(person):
    #load data
    X_train, y_train, X_test, y_test = loadPerson(
            person = person,
            classFunc = classFunc,
            featureFunc = featureFunc
    )
        
    #init academic loop to optimize k param
    k = 1
    anova_filter = SelectKBest(f_regression)
    lda          = LinearDiscriminantAnalysis()
    anova_lda    = Pipeline([
        ('anova', anova_filter), 
        ('lda', lda)
    ])
    anova_lda.set_params(anova__k=k)

    K_CV = KFold(n=len(X_train), 
        n_folds=len(X_train),
        random_state=17, #fixed randomseed ensure that the sets are always the same
        shuffle=False
    ) #leave out one validation

    predictions, truths = [], []
    for train_index, CV_index in K_CV: #train index here is a part of the train set
        #train
        anova_lda.fit(X_train[train_index], y_train[train_index])

        #predict
        pred = anova_lda.predict(X_train[CV_index])

        #save for metric calculations
        predictions.extend(pred)
        truths.extend(y_train[CV_index])

    #optimization metric:
    best_acc = UT.accuracy(predictions, truths)
    best_k   = k
    
    #now try different k values
    for k in range(2,len(X_train[0])):
        anova_filter = SelectKBest(f_regression)
        lda          = LinearDiscriminantAnalysis()
        anova_lda    = Pipeline([
            ('anova', anova_filter), 
            ('lda', lda)
        ])
        #set k param
        anova_lda.set_params(anova__k=k)

        #leave one out validation to determine how good the k value performs
        K_CV = KFold(n=len(X_train), 
            n_folds=len(X_train),
            random_state=17, #fixed randomseed ensure that the sets are always the same
            shuffle=False
        )

        predictions, truths = [], []
        for train_index, CV_index in K_CV: #train index here is a part of the train set
            #train
            anova_lda.fit(X_train[train_index], y_train[train_index])

            #predict
            pred = anova_lda.predict(X_train[CV_index])

            #save for metric calculations
            predictions.extend(pred)
            truths.extend(y_train[CV_index])

        #optimization metric:
        curr_acc = UT.accuracy(predictions, truths)
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_k   = k

    #now the k param is optimized and stored in best_k

    #create classifier and train it on all train data
    anova_filter = SelectKBest(f_regression)
    lda          = LinearDiscriminantAnalysis()
    anova_lda    = Pipeline([
        ('anova', anova_filter), 
        ('lda', lda)
    ])
    #set k param
    anova_lda.set_params(anova__k=best_k)
    anova_lda.fit(X_train, y_train)

    predictions = anova_lda.predict(X_test)

    acc  = UT.accuracy(predictions, y_test)
    (tpr,tnr,fpr,fnr) = UT.tprtnrfprfnr(predictions, y_test)
    auc = UT.auc(predictions, y_test)

    print('person: ', person, 
        ' - k: '  , str(best_k),
        ' - acc: ', str(acc),
        ' - tpr: ' , str(tpr),
        ' - tnr: ' , str(tnr),
        ' - auc: ', str(auc),
        'used features', anova_lda.named_steps['anova'].get_support()
    )
    return [best_k, acc,tpr,tnr,fpr,fnr,auc]


if __name__ == '__main__':
    #multithreaded
    pool = Pool(processes=8)
    results = pool.map( PersonWorker, range(1,33) )
    pool.close()
    pool.join()

    results = np.array(results)
    #results = lest<[best_k, acc, tpr, tnr, fpr, fnr, auc]>
    print(
        'avg acc', np.average(results[:,1]),
        'avg auc', np.average(results[:,6])
    )