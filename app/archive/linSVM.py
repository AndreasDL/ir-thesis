import datetime
import time
from multiprocessing import Pool

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC

import dataLoader as DL
import util as UT
from archive import featureExtractor as FE


def featureFunc(samples):
    features = []
    features.extend(FE.LMinRFraction(samples,intervalLength=63, overlap=0))
    features.extend(FE.FrontlineMidlineThetaPower(samples, channels=['Cz','Fz'], intervalLength=63, overlap=0))


    return np.array(features)

#lin svm, no csp , C optimization 
def PersonWorker(person):
    print('starting on person: ', str(person))

    #data = 40 videos x 32 alpha(csp channel)
    X_train, y_train, X_test, y_test, csp = DL.loadPerson(person=person,
        featureFunc = featureFunc,
        use_csp=False,
        use_median = False
    )
    
    C = 1
    clf = LinearSVC(C=C,random_state=40)
    K_CV = KFold(n=len(X_train), n_folds=len(X_train), random_state=17, shuffle=False) #leave out one validation
    predictions, truths = [], []
    for train_index, CV_index in K_CV:
        #train
        clf.fit(X_train[train_index], y_train[train_index])

        #predict
        pred = clf.predict(X_train[CV_index])

        #save for metric calculations
        predictions.extend(pred)
        truths.extend(y_train[CV_index])

    #optimization metric:
    best_metric = UT.auc(predictions, truths)
    best_C = C
    
    #try other C values
    for C in [0.01,0.03,0.1,0.3,3,10]:
        clf = LinearSVC(C=C,random_state=40)
        K_CV = KFold(n=len(X_train), n_folds=len(X_train), random_state=17, shuffle=True) #leave out one validation
        predictions, truths = [], []
        for train_index, CV_index in K_CV:
            #train
            clf.fit(X_train[train_index], y_train[train_index])

            #predict
            pred = clf.predict(X_train[CV_index])

            #save for metric calculations
            predictions.extend(pred)
            truths.extend(y_train[CV_index])

        #optimization metric:
        metric = UT.auc(predictions, truths)
        if metric > best_metric:
            best_metric = metric
            best_C = C

    #C param is now optimized, its value is stored in best_C

    #calculate all performance metrics on testset, using the optimal classifier
    clf = LinearSVC(C=C,random_state=40)
    clf.fit(X_train,y_train) #fit all training data
    #print("coef ", clf.coef_)
    predictions = clf.predict(X_test)

    acc  = UT.accuracy(predictions, y_test)
    (tpr,tnr,fpr,fnr) = UT.tprtnrfprfnr(predictions, y_test)
    auc = UT.auc(predictions, y_test)

    print('person: ', person, 
        ' - acc: ', str(acc),
        ' - tpr: ' , str(tpr),
        ' - tnr: ' , str(tnr),
        ' - auc: ', str(auc)
    )

    return [acc,tpr,tnr,fpr,fnr,auc]

if __name__ == "__main__":
    #PersonWorker(1)
    #exit()


    #multithreaded
    pool = Pool(processes=8)
    results = pool.map( PersonWorker, range(1,33) )
    pool.close()
    pool.join()

    results = np.array(results)
    #results = lest<[channelPairs, acc, tpr, tnr, fpr, fnr, auc]>
    print('avg acc:', np.average(results[:,0]), 'avg auc:', np.average(results[:,5]))

    #output    
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    f = open("output" + str(st) + ".txt", 'w')
    f.write('person;channel pairs;acc;tpr;tnr;fpr;fnr;auc\n')

    for person, result in enumerate(results):
        (acc, tpr, tnr, fpr, fnr, auc) = result

        f.write(str(person+1) + ';' + 
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