import time
import pickle
import datetime
import util as UT
from csp import Csp
import numpy as np
import dataLoader as DL
import featureExtractor as FE
from multiprocessing import Pool
from sklearn.cross_validation import KFold, StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def featureFunc(samples):
    features = []
    features.extend(FE.alphaPowers(samples))

    return np.array(features)

def PersonWorker(person):
    print('starting on person: ', str(person))

    #data = 40 videos x 32 alpha(csp channel)
    X_train, y = DL.loadPerson(person=person,
        featureFunc = featureFunc,
        use_median=True
    )

    #optimize channelPairs with leave-one out validation
    #prior probabilities
    pos_prior = np.sum(y)
    neg_prior = 40 - pos_prior
    pos_prior /= float(40)
    neg_prior /= float(40)

    #academic loop start with 1 channelPair
    channelPairs = 1

    #filter out the channel pairs
    X = np.zeros((len(X_train),channelPairs * 2,))
    top_offset = channelPairs * 2 - 1
    for j, k in zip(range(channelPairs), range(31,31-channelPairs,-1)):
        X[:,j] = X_train[:,j]
        X[:,top_offset -j] = X_train[:,k]

    #LDA
    lda = LinearDiscriminantAnalysis(priors=[neg_prior, pos_prior])
    K_CV = KFold(n=len(X), n_folds=len(X), random_state=17, shuffle=False) #leave out one validation
    predictions, truths = [], []
    for train_index, CV_index in K_CV:
        #train
        lda = lda.fit(X[train_index], y[train_index])

        #predict
        pred = lda.predict(X[CV_index])

        #save for metric calculations
        predictions.extend(pred)
        truths.extend(y[CV_index])

    #optimization metric:
    best_metric = UT.accuracy(predictions, truths)
    best_channelPairs = channelPairs
    best_predictions = predictions
    best_truths = truths

    
    #try other channel pairs
    for channelPairs in range(2,17):
        #filter out the channel pairs
        X = np.zeros((len(X_train),channelPairs * 2,))
        top_offset = channelPairs * 2 - 1
        for j, k in zip(range(channelPairs), range(31,31-channelPairs,-1)):
            X[:,j] = X_train[:,j]
            X[:,top_offset -j] = X_train[:,k]

        #LDA
        lda = LinearDiscriminantAnalysis(priors=[neg_prior, pos_prior])
        K_CV = KFold(n=len(X), n_folds=len(X), random_state=17, shuffle=True) #leave out one validation
        predictions, truths = [], []
        for train_index, CV_index in K_CV:
            #train
            lda = lda.fit(X[train_index], y[train_index])

            #predict
            pred = lda.predict(X[CV_index])

            #save for metric calculations
            predictions.extend(pred)
            truths.extend(y[CV_index])

        #optimization metric:
        metric = UT.accuracy(predictions, truths)
        if metric > best_metric:
            best_metric = metric
            best_channelPairs = channelPairs
            best_predictions = predictions
            best_truths = truths

    #channel pairs are now optimized, its value is stored in best_channelPairs

    #calculate all performance metrics on testset, using the optimal classifier
    acc  = UT.accuracy(best_predictions, best_truths)
    (tpr,tnr,fpr,fnr) = UT.tprtnrfprfnr(best_predictions, best_truths)
    auc = UT.auc(best_predictions, best_truths)

    print('person: ', person, 
        ' - channelPairs: ', str(best_channelPairs),
        ' - acc: ', str(acc),
        ' - tpr: ' , str(tpr),
        ' - tnr: ' , str(tnr),
        ' - auc: ', str(auc)
    )

    return [best_channelPairs, acc,tpr,tnr,fpr,fnr,auc]

if __name__ == "__main__":
    #multithreaded
    pool = Pool(processes=8)
    results = pool.map( PersonWorker, range(1,33) )
    pool.close()
    pool.join()

    results = np.array(results)
    #results = lest<[channelPairs, acc, tpr, tnr, fpr, fnr, auc]>
    print('avg acc:', np.average(results[:,1]), 'avg auc:', np.average(results[:,6]))

    #output    
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    f = open("output" + str(st) + ".txt", 'w')
    f.write('person;channel pairs;acc;tpr;tnr;fpr;fnr;auc\n')

    for person, result in enumerate(results):
        (channelPairs, acc, tpr, tnr, fpr, fnr, auc) = result

        f.write(str(person+1) + ';' + 
            str(channelPairs) + ';' +
            str(acc) + ';' + 
            str(tpr) + ';' + str(tnr) + ';' +
            str(fpr) + ';' + str(fnr) + ';' +
            str(auc) + '\n'
        )

    f.write('\nmedian;')
    for column in range(len(results[0])):
        f.write(str(np.median(results[:,column])) + ';')

    f.write('\navg;')
    for column in range(len(results[0])):
        f.write(str(np.average(results[:,column])) + ';')

    f.write('\nstd;')
    for column in range(len(results[0])):
        f.write(str(np.std(results[:,column])) + ';')

    f.close()