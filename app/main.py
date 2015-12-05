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
    pos_count = np.sum(y)
    neg_count = 40 - pos_count

    #LDA
    lda = LinearDiscriminantAnalysis(priors=[neg_count, pos_count])
    
    K_CV = KFold(len(X), n_folds=CVSets, random_state=17, shuffle=True)
    acc, tp, tn , fp , fn , auc = 0, 0, 0, 0, 0, 0
    
    #for auc
    predictions, truths = [], []
    for train_index, CV_index in K_CV:
        #train
        lda = lda.fit(X[train_index], y[train_index])

        #predict
        pred = lda.predict(X[CV_index])
        #save for auc 
        predictions.extend(pred)
        truths.append(y[CV_index])
        
        #MSE train err
        (ttp,ttn,tfp,tfn) = UT.tptnfpfn(pred, y[CV_index])
        tp  += ttp
        tn  += ttn
        fp  += tfp
        fn  += tfn
        acc += UT.accuracy(pred, y[CV_index])

    #accuracy
    acc /= float(CVSets)
    
    #tptnfpfn => to rates
    tp /= pos_count
    tn /= neg_count
    fp /= pos_count
    fn /= neg_count

    #auc
    auc = UT.auc(predictions, truths)

    print('person: ', person, 
        ' - acc: ', str(acc),
        ' - tp: ' , str(tp),
        ' - tn: ' , str(tn),
        ' - auc: ', str(auc)
    )

    return [acc,tp,tn,fp,fn,auc]

if __name__ == "__main__":
    #multithreaded
    pool = Pool(processes=1)
    results = pool.map( PersonWorker, range(1,33) )
    pool.close()
    pool.join()

    results = np.array(results)
    print('avg acc:', np.average(results[:,0]), 'avg auc:', np.average(results[:,5]))


    #output    
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    f = open("output" + str(st) + ".txt", 'w')
    f.write('person;acc;tp;tn;fp;fn;auc\n')

    for person, result in enumerate(results):
        (acc, tp, tn, fp, fn, auc) = result

        f.write(str(person+1) + ';' + str(acc) + ';' + 
            str(tp) + ';' + str(tn) + ';' +
            str(fp) + ';' + str(fn) + ';' +
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