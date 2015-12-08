import time
import pickle
import datetime
import util as UT
import numpy as np
import dataLoader as DL
import featureExtractor as FE
from multiprocessing import Pool
from sklearn.cross_validation import KFold,StratifiedShuffleSplit
from sklearn.svm import LinearSVC

def featureFunc(samples):
    features = []
    features.extend(FE.LMinRFraction(samples,intervalLength=63, overlap=0))
    features.extend(FE.FrontlineMidlineThetaPower(samples, channels=['Cz','Fz'], intervalLength=63, overlap=0))


    return np.array(features)

def PersonWorker(person):
    print('starting on person: ', str(person))

    #data = 40 videos x 32 alpha(csp channel)
    data, labels = DL.loadPerson(person=person,
        featureFunc = featureFunc,
        use_csp=False,
        use_median = False
    )

    #break off test set
    sss = StratifiedShuffleSplit(labels, n_iter=10, test_size=0.25, random_state=19)
    for train_set_index, test_set_index in sss:
        X, y = data[train_set_index], labels[train_set_index]
        X_test, y_test  = data[test_set_index], labels[test_set_index]
        break;

    
    C = 1
    clf = LinearSVC(C=C,random_state=40)
    K_CV = KFold(n=len(X), n_folds=len(X), random_state=17, shuffle=False) #leave out one validation
    predictions, truths = [], []
    for train_index, CV_index in K_CV:
        #train
        clf.fit(X[train_index], y[train_index])

        #predict
        pred = clf.predict(X[CV_index])

        #save for metric calculations
        predictions.extend(pred)
        truths.extend(y[CV_index])

    #optimization metric:
    best_metric = UT.auc(predictions, truths)
    best_C = C
    
    #try other channel pairs
    for C in [0.01,0.03,0.1,0.3,3,10]:


        #LDA
        clf = LinearSVC(C=C,random_state=40)
        K_CV = KFold(n=len(X), n_folds=len(X), random_state=17, shuffle=True) #leave out one validation
        predictions, truths = [], []
        for train_index, CV_index in K_CV:
            #train
            clf.fit(X[train_index], y[train_index])

            #predict
            pred = clf.predict(X[CV_index])

            #save for metric calculations
            predictions.extend(pred)
            truths.extend(y[CV_index])

        #optimization metric:
        metric = UT.auc(predictions, truths)
        if metric > best_metric:
            best_metric = metric
            best_C = C

    #channel pairs are now optimized, its value is stored in best_channelPairs

    #calculate all performance metrics on testset, using the optimal classifier
    clf = LinearSVC(C=C,random_state=40)
    clf.fit(X,y) #fit all training data
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