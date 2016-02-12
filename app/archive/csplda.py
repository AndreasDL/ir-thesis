import datetime
import time
from multiprocessing import Pool

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import dataLoader as DL
import util as UT
from archive import featureExtractor as FE


def featureFunc(samples):
    features = []
    features.extend(FE.alphaPowers(samples))
    #features.extend(FE.betaPowers(samples))
    #features.extend(FE.gammaPowers(samples))
    #features.extend(FE.deltaPowers(samples))
    #features.extend(FE.thetaPowers(samples))

    return np.array(features)

#csp+lda + cahnnelpair optimization
def PersonWorker(person):
    print('starting on person: ', str(person))

    #data = 40 videos x 32 alpha(csp channel)
    X_train, y_train, X_test, y_test, csp = DL.loadPerson(person=person,
        featureFunc = featureFunc,
        use_median=False,
        use_csp=True,
        prefilter=False
    )

    #store weights of upper CSP channel for topoplots
    csp.write_filters()

    #optimize channelPairs with leave-one out validation
    #prior probabilities
    pos_prior = np.sum(y_train)
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
        lda = lda.fit(X[train_index], y_train[train_index])

        #predict
        pred = lda.predict(X[CV_index])

        #save for metric calculations
        predictions.extend(pred)
        truths.extend(y_train[CV_index])

    #optimization metric:
    best_metric = UT.accuracy(predictions, truths)
    best_channelPairs = channelPairs
    
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
            lda = lda.fit(X[train_index], y_train[train_index])

            #predict
            pred = lda.predict(X[CV_index])

            #save for metric calculations
            predictions.extend(pred)
            truths.extend(y_train[CV_index])

        #optimization metric:
        metric = UT.accuracy(predictions, truths)
        if metric > best_metric:
            best_metric = metric
            best_channelPairs = channelPairs

    #channel pairs are now optimized, its value is stored in best_channelPairs

    #calculate all performance metrics on testset, using the optimal classifier
    lda = LinearDiscriminantAnalysis(priors=[neg_prior, pos_prior])
    lda = lda.fit(X_train,y_train) #fit all training data
    predictions = lda.predict(X_test)

    acc  = UT.accuracy(predictions, y_test)
    (tpr,tnr,fpr,fnr) = UT.tprtnrfprfnr(predictions, y_test)
    auc = UT.auc(predictions, y_test)

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