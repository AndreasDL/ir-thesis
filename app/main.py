import os
import time
import pickle
import datetime
import util as UT
import numpy as np
import dataLoader as DL
import featureExtractor as FE
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




if __name__ == "__main__":
    CVSets = float(4)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    f = open("output" + str(st) + ".txt", 'w')
    f.write('CVSets: ' + str(CVSets) + '\n' +
        'person;acc;tp;tn;fp;fn;auc\n'
    )

    for person in range(1,33):
        #split using median, use all data
        X, y = DL.loadPerson(person, 
            featureFunc=FE.alphaPowers,
            preprocessFunc=UT.csp
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