import time
import pickle
import datetime
import util as UT
import numpy as np
import dataLoader as DL
import featureExtractor as FE
from multiprocessing import Pool
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def featureFunc(samples):
    features = []
    features.extend(FE.alphaPowers(samples))
    #features.extend(FE.betaPowers(samples))
    #features.extend(FE.deltaPowers(samples))
    #features.extend(FE.gammaPowers(samples))
    #features.extend(FE.thetaPowers(samples))

    return np.array(features)

def PersonWorker(person):
    print('starting on person: ', str(person))
    CVSets = 40 #leave one out validation

    #split using median, use all data
    X, y = DL.loadPerson(person=person, 
        featureFunc=featureFunc,
        preprocessFunc=UT.csp,
        use_median=False
    )
    pos_prior = np.sum(y)
    neg_prior = 40 - pos_prior
    
    #LDA
    lda = LinearDiscriminantAnalysis(priors=[neg_prior/float(40), pos_prior/float(40)])

    K_CV = KFold(len(X), n_folds=CVSets, random_state=17, shuffle=True)
    predictions, truths = [], []
    for train_index, CV_index in K_CV:
        #train
        lda = lda.fit(X[train_index], y[train_index])

        #predict
        pred = lda.predict(X[CV_index])

        #save for metric calculations
        predictions.extend(pred)
        truths.extend(y[CV_index])

    #accuracy
    acc = UT.accuracy(predictions, truths)
    
    #tptnfpfn => to rates
    (tpr,tnr,fpr,fnr) = UT.tprtnrfprfnr(predictions, truths)

    #auc
    auc = UT.auc(predictions, truths)

    print('person: ', person, 
        ' - acc: ', str(acc),
        ' - tpr: ' , str(tpr),
        ' - tnr: ' , str(tnr),
        ' - auc: ', str(auc)
    )

    return [acc,tpr,tnr,fpr,fnr,auc]

if __name__ == "__main__":
    #multithreaded
    pool = Pool(processes=8)
    results = pool.map( PersonWorker, range(1,33) )
    pool.close()
    pool.join()

    results = np.array(results)
    print('avg acc:', np.average(results[:,0]), 'avg auc:', np.average(results[:,5]))

    #output    
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    f = open("output" + str(st) + ".txt", 'w')
    f.write('person;acc;tpr;tnr;fpr;fnr;auc\n')

    for person, result in enumerate(results):
        (acc, tpr, tnr, fpr, fnr, auc) = result

        f.write(str(person+1) + ';' + str(acc) + ';' + 
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