import datetime
import time
from multiprocessing import Pool

import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

import archive.util as UT
import personLoader as DL
from archive import featureExtractor as FE


def featureFunc(samples):
    features = []
    features.extend(FE.powers(samples,'alpha'))
    features.extend(FE.powers(samples,'delta'))
    features.extend(FE.powers(samples,'gamma'))
    features.extend(FE.powers(samples,'beta'))
    #features.extend(FE.powers(samples,'alpha'))#todo lower and upper beta

    return np.array(features)

def PersonWorker(person):
    print('starting on person: ', str(person))

    #data = 40 videos x 32 alpha(csp channel)
    (X_train, y_train, X_test, y_test) = DL.loadPersonEpochDimRedu(person=person,
        featureFunc = featureFunc,
    )
    
    #http://stackoverflow.com/questions/26963454/lda-ignoring-n-components => only 1 feature :(
    print(np.shape(X_train))

    svm = LinearSVC()
    svm.fit(X_train, y_train)
    
    y = svm.predict(X_train)
    y = label_binarize(y, classes=[0, 1, 2, 3])
    train_auc = UT.auc(y, y_train)

    y = svm.predict(X_test)
    y = label_binarize(y, classes=[0, 1, 2, 3])
    test_auc = UT.auc(y, y_test)


    print('person: ', person, 
        ' - train auc: ', str(train_auc),
        ' - test auc: ' , str(test_auc)
    )

    return [train_auc, test_auc]

if __name__ == "__main__":

    #multithreaded
    pool = Pool(processes=1)
    results = pool.map( PersonWorker, range(1,33) )
    pool.close()
    pool.join()

    results = np.array(results)
    #results = lest<[channelPairs, acc, tpr, tnr, fpr, fnr, auc]>
    print('avg train_auc:', np.average(results[:,0]), 'avg test_auc:', np.average(results[:,1]))

    #output    
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    f = open("output" + str(st) + ".txt", 'w')
    f.write('person;train_acc;test_auc\n')

    for person, result in enumerate(results):
        (train_auc, test_auc) = result

        f.write(str(person+1) + ';' + 
            str(train_auc) + ';' +
            str(test_auc)  + ';\n'
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