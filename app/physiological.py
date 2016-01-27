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

from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 30}
matplotlib.rc('font', **font)


channelNames = {
#after removing EEG channels
    'GSR'  : 0, #(values from Twente converted to Geneva format (Ohm))
    'Respiration belt' : 1,
    'Plethysmograph' : 2,
    'Temperature' : 3
}
Fs = 128 #samples have freq 128Hz


def plot(data, descr="value"):
    n = len(data)
    plt.plot(np.arange(n)/128, data, 'r-')
    plt.xlabel('time (s)')
    plt.ylabel(descr)
    plt.show()
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
def featureFunc(data):
    #filter out right channels
    samples = np.array(data['data'])[:,36:,:] #throw out EEG channels & eye and muscle movement
    
    #lowpass filter => sufficient smoothing is needed to extract heart rate from plethysmograph
    Fs = 128 #samples have freq 128Hz
    n = len(samples[0])#8064 #number of samples
    nyq  = 0.5 * Fs 
    low  = 3 / nyq #lower values => smoother
    b, a = butter(6, low, btype='low')
    for video in range(len(samples)):
        for channel in range(len(samples[video])):
            samples[video][channel] = lfilter(b, a, samples[video][channel])

    #extract features
    features = []
    for video in samples:
        video_features = []

        for channel in video:
            #minVal = np.min( channel )
            #maxVal = np.max( channel )

            video_features.append( np.mean(  channel) )
            video_features.append( np.std(   channel) )
            #video_features.append( np.median(channel) )
            #video_features.append( minVal )
            #video_features.append( maxVal )
            #video_features.append( maxVal - minVal )
        
        video_features.extend(
            heartStuffz(
                video[channelNames['Plethysmograph']]
            )
        )

        features.append(video_features)
    #features look like this (person specific)
    #list[video] = | avg_GSR | std_GSR | avg_rb  | std_rb  |
    #              | avg_ply | std_ply | avg_tem | std_tem |
    #              | avg_hr  | var_interbeats |
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

    print(X_train)
    exit;
        
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
    pool = Pool(processes=1)
    results = pool.map( PersonWorker, range(1,2) )
    pool.close()
    pool.join()

    results = np.array(results)
    #results = lest<[best_k, acc, tpr, tnr, fpr, fnr, auc]>
    print(
        'avg acc', np.average(results[:,1]),
        'avg auc', np.average(results[:,6])
    )