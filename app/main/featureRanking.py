from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import personLoader
import Classificators
import featureExtractor as FE
from sklearn.cross_validation import KFold
from scipy.stats import pearsonr
import numpy as np
import datetime
import time
import featureExtractor
from featureExtractor import all_channels

from personLoader import load, dump
from multiprocessing import Pool
POOL_SIZE = 5

def getFeatures():
    # create the features
    featExtr = FE.MultiFeatureExtractor()

    #physiological signals
    for channel in FE.all_phy_channels:
        featExtr.addFE(FE.AvgExtractor(channel, ''))
        featExtr.addFE(FE.STDExtractor(channel, ''))

    featExtr.addFE(FE.AVGHeartRateExtractor())
    featExtr.addFE(FE.STDInterBeatExtractor())

    #EEG
    for channel in FE.all_EEG_channels:
        featExtr.addFE(
            FE.AlphaBetaExtractor(
                channels=[channel],
                featName='A/B ' + FE.all_channels[channel]
            )
        )

        '''
        featExtr.addFE(
            FE.FrontalMidlinePower(
                channels=[channel],
                featName="FM " + FE.all_channels[channel]
            )
        )
        '''

        for freqband in FE.startFreq:
            featExtr.addFE(
                FE.DEExtractor(
                    channels=[channel],
                    freqBand=freqband,
                    featName='DE ' + FE.all_channels[channel] + '(' + freqband + ')'
                )
            )

            featExtr.addFE(
                FE.PSDExtractor(
                    channels=[channel],
                    freqBand=freqband,
                    featName='PSD ' + FE.all_channels[channel] + '(' + freqband + ')'
                )
            )


    for left, right in zip(FE.all_left_channels, FE.all_right_channels):
        featExtr.addFE(
            FE.LMinRLPlusRExtractor(
                left_channels=[left],
                right_channels=[right],
                featName='LR ' + FE.all_channels[left] + ',' + FE.all_channels[right]
            )
        )

        for freqband in FE.startFreq:
            featExtr.addFE(
                FE.DASMExtractor(
                    left_channels=[left],
                    right_channels=[right],
                    freqBand=freqband,
                    featName='DASM ' + FE.all_channels[left] + ',' + FE.all_channels[right]
                )
            )

            featExtr.addFE(
                FE.RASMExtractor(
                    left_channels=[left],
                    right_channels=[right],
                    freqBand=freqband,
                    featName='RASM ' + FE.all_channels[left] + ',' + FE.all_channels[right]
                )
            )

    for front, post in zip(FE.all_frontal_channels, FE.all_posterior_channels):
        for freqband in FE.startFreq:
            featExtr.addFE(
                FE.DCAUExtractor(
                    frontal_channels=[front],
                    posterior_channels=[post],
                    freqBand=freqband,
                    featName='DCAU ' + FE.all_channels[front] + ',' + FE.all_channels[post]
                )
            )

            featExtr.addFE(
                FE.RCAUExtractor(
                    frontal_channels=[front],
                    posterior_channels=[post],
                    freqBand=freqband,
                    featName='RCAU ' + FE.all_channels[front] + ',' + FE.all_channels[post]
                )
            )

    return featExtr
def accuracy(predictions, truths):
    acc = 0
    for pred, truth in zip(predictions, truths):
        acc += (pred == truth)

    return acc / float(len(predictions))

def getPersonRankings(person):
    #load all features & keep them in memory
    y_cont = load('cont_y_p' + str(person))
    if y_cont == None:
        print('[Warn] Rebuilding cache -  person ' + str(person))
        classificator = Classificators.ContValenceClassificator()
        featExtr = getFeatures()
        personLdr = personLoader.NoTestsetLoader(classificator, featExtr)

        X, y_cont = personLdr.load(person)

        dump(X,'X_p' + str(person))
        dump(y_cont,'cont_y_p' + str(person))
    else:
        X = load('X_p' +str(person))

    X = np.array(X)
    y_cont = np.array(y_cont)
    y_disc = y_cont
    y_disc[ y_disc <= 5 ] = 0
    y_disc[ y_disc >  5 ] = 1

    for index,val in enumerate(np.std(X,axis=0)):
        if val == 0:
            print('warning zero std for feature index: ', index, ' (', personLoader.featureExtractor.getFeatureNames()[index])

    #manual Feature standardization
    X = X - np.average(X,axis=0)
    X = np.true_divide(X, np.std(X,axis=0) )

    #statistical tests
    #get pearson
    corr = []
    for index in range(len(X[0])):
        corr.append( pearsonr(X[:, index], y_cont) )

    #model based:
    #normal regression
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X, y_cont)
    lr_scores = lr.coef_

    #l1 regression
    alphas = [0.03,0.1,0.3,1,3,10]
    best_alpha = 0.01
    best_acc = 0
    for train_index, test_index in KFold(len(y_cont), n_folds=5):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_cont[train_index], y_cont[test_index]

        lasso = Lasso(alpha=best_alpha)
        lasso.fit(X_train,y_train)
        pred = lasso.predict(X_test)
        best_acc += accuracy(pred,y_cont)
    best_acc /= float(5)

    for alpha in alphas:
        acc = 0
        for train_index, test_index in KFold(len(y_cont), n_folds=5):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_cont[train_index], y_cont[test_index]

            lasso = Lasso(alpha=alpha)
            lasso.fit(X_train,y_train)
            pred = lasso.predict(X_test)
            acc += accuracy(pred,y_test)

        acc /= float(5)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X, y_cont)
    l1_scores = lasso.coef_

    #l2 regression
    alphas = [0.03,0.1,0.3,1,3,10]
    best_alpha = 0.01
    best_acc = 0
    for train_index, test_index in KFold(len(y_cont), n_folds=5):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_cont[train_index], y_cont[test_index]

        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_train,y_train)
        pred = ridge.predict(X_test)
        best_acc += accuracy(pred,y_test)
    best_acc /= float(5)

    for alpha in alphas:
        acc = 0
        for train_index, test_index in KFold(len(y_cont), n_folds=5):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_cont[train_index], y_cont[test_index]

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train,y_train)
            pred = lasso.predict(X_test)
            acc += accuracy(pred,y_test)

        acc /= float(5)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X, y_cont)
    l2_scores = ridge.coef_

    #svm coefficients
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y_disc)
    svm_weights = (clf.coef_ ** 2).sum(axis=0)
    svm_weights /= float(svm_weights.max())

    #rf importances
    #grow forest
    forest = RandomForestClassifier(
        n_estimators=3000,
        max_features='auto',
        criterion='gini',
        n_jobs=-1,
    )
    forest.fit(X,y_disc)
    #get importances
    importances = forest.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)

    #optimization problem
    #coordinated search111

    # [pearson_r, mutual inf, max inf, dist, l1 coef, l2 coef, svm coef, rf importances, coord search]
    #featExtr = getFeatures()
    #featnames = featExtr.getFeatureNames()
    pers_results = []
    for corr, lr_scores, l1_scores, l2_scores, svm_weights, importances in zip(corr, lr_scores, l1_scores, l2_scores, svm_weights, importances):
        pers_results.append([corr[0], lr_scores, l1_scores, l2_scores, svm_weights, importances])

    return pers_results#, featExtr.featureExtrs
def top30Accs(results):
    #results = list([corr[0], lr_scores, l1_scores, l2_scores, svm_weights, importances])

    #take averages
    avg_results = [[0] * len(results[0,0])] * len(results[0])

    for result in results:
        for i in range(len(result)): #each feature
            for j in range(len(result[i])):
                avg_results[i][j] += result[i][j]
        avg_results = np.array(avg_results)
        avg_results = np.true_divide(avg_results,float(len(results)))

    acc_no_features = []
    for person in range(1,len(results)+1):
        #load all features & keep them in memory
        y_cont = load('cont_y_p' + str(person))
        X = load('X_p' +str(person))

        X = np.array(X)
        y_cont = np.array(y_cont)
        y_disc = y_cont
        y_disc[ y_disc <= 5 ] = 0
        y_disc[ y_disc >  5 ] = 1

        for index,val in enumerate(np.std(X,axis=0)):
            if val == 0:
                print('warning zero std for feature index: ', index, ' (', personLoader.featureExtractor.getFeatureNames()[index])

        #manual Feature standardization
        X = X - np.average(X,axis=0)
        X = np.true_divide(X, np.std(X,axis=0) )

        #6 accs on 6 metrics
        pers_accs = []
        for i in range(len(avg_results[0])):
            #sort features
            indices = np.array(np.argsort(avg_results[:,i])[::-1])

            #get first 30 indexes
            features_to_keep = indices[:30]
            X_temp = X[:,features_to_keep]

            model_accs = []
            for i in range(1,len(features_to_keep)+1):
                X_temp = X[:,:i]

                #svm coefficients
                clf = svm.SVC()
                clf.fit(X_temp, y_disc)

                pred = clf.predict(X_temp)
                acc = accuracy(pred,y_disc)

                model_accs.append(acc)
            pers_accs.append(model_accs)

        acc_no_features.append(pers_accs)

    return acc_no_features

def genReport(results):
    what_mapper = {
        featureExtractor.PSDExtractor: 'PSD',
        featureExtractor.DEExtractor: 'DE',
        featureExtractor.RASMExtractor: 'RASM',
        featureExtractor.DASMExtractor: 'DASM',
        featureExtractor.AlphaBetaExtractor: 'AB',
        featureExtractor.DCAUExtractor: 'DCAU',
        featureExtractor.RCAUExtractor: 'RCAU',
        featureExtractor.LMinRLPlusRExtractor: 'LminR',
        featureExtractor.FrontalMidlinePower: 'FM',
        featureExtractor.AvgExtractor: 'AVG',
        featureExtractor.STDExtractor: 'STD',
        featureExtractor.AVGHeartRateExtractor: 'AVG HR',
        featureExtractor.STDInterBeatExtractor: 'STD HR'
    }

    #take averages
    avg_results = []
    for person in results:
        for i in range(len(person)):
            avg_results.append([0,0,0,0,0,0])
            for j in range(len(person[i])):
                avg_results[i][j] += person[i][j]
    avg_results = np.array(avg_results)
    avg_results = np.true_divide(avg_results,float(len(results)))

    #output to file
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
    f = open('../../results/ranking_valence' + str(st) + ".csv", 'w')

    f.write('featname;eeg/phy;what;channels;waveband;pearson_r;lr coef;l1 coef;l2 coef;svm coef;rf importances;\n')
    for featExtr, result in zip(getFeatures().featureExtrs, avg_results):

        feat_name = featExtr.featureName
        feat_what = what_mapper[type(featExtr)]
        feat_eeg, feat_channel, feat_waveband = None, None, None

        if type(featExtr) in [featureExtractor.AVGHeartRateExtractor, featureExtractor.AvgExtractor, featureExtractor.STDExtractor, featureExtractor.STDInterBeatExtractor]:
            feat_eeg = 'phy'
            feat_waveband = 'n/a' #no freq bands
            feat_channel = 'n/a'
        else:
            feat_eeg = 'eeg'

            #single channel
            if type(featExtr) in [featureExtractor.PSDExtractor, featureExtractor.DEExtractor, featureExtractor.FrontalMidlinePower]:
                feat_waveband = featExtr.usedFeqBand
                feat_channel = all_channels[featExtr.usedChannelIndexes[0]]

            elif type(featExtr) == featureExtractor.AlphaBetaExtractor:
                feat_waveband = 'alpha & beta'
                feat_channel = all_channels[featExtr.usedChannelIndexes[0]]

            elif type(featExtr) == featureExtractor.LMinRLPlusRExtractor:
                feat_waveband = 'alpha'
                feat_channel = [
                    all_channels[featExtr.left_channels[0]],
                    all_channels[featExtr.right_channels[0]]
                ]

            #multiple channels Left and right
            elif type(featExtr) in [featureExtractor.DASMExtractor, featureExtractor.RASMExtractor]:
                feat_waveband = featExtr.leftDEExtractor.usedFeqBand
                feat_channel = [
                    all_channels[featExtr.leftDEExtractor.usedChannelIndexes[0]],
                    all_channels[featExtr.rightDEExtractor.usedChannelIndexes[0]]
                ]

            #multiple channels post and front
            else:
                feat_waveband = featExtr.frontalDEExtractor.usedFeqBand
                feat_channel = [
                    all_channels[featExtr.frontalDEExtractor.usedChannelIndexes[0]],
                    all_channels[featExtr.posteriorDEExtractor.usedChannelIndexes[0]]
                ]

        #f.write('featname;eeg/phy;what;channels;waveband;pearson_r;l1 coef;l2 coef;svm coef;rf importances;\n')
        f.write(
            str(feat_name) + ';' +
            str(feat_eeg) + ';' +
            str(feat_what) + ';' +
            str(feat_channel) + ';' +
            str(feat_waveband) + ';'
        )

        for metric in result:
            f.write(str(abs(metric)) + ";")

        f.write("\n")

    f.close()
def accReport(accs):
    #output to file
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
    f = open('../../results/accs_svm_valence' + str(st) + ".csv", 'w')

    #averages
    avg_accs = np.average(accs,axis=0)
    avg_accs = np.transpose(avg_accs)

    #write averages
    f.write('number of features;pearson;lr;l1;l2;svm;rf\n')
    for line in avg_accs:
        for acc in line:
            f.write(str(acc) + ';')
        f.write("\n")

    f.write("\n")
    f.write("\n")

    #write best accs per person
    max_accs = np.amax(accs, axis=2)
    f.write("best accs obtained for each person")
    f.write('person;pearson;lr;l1;l2;svm;rf\n')
    for person, line in enumerate(avg_accs):
        f.write(str(person+1) + ";" )
        for acc in line:
            f.write(str(acc) + ';')
        f.write("\n")

    f.close()

if __name__ == '__main__':
    stop_person = 33
    if stop_person < 33:
        print('[warn] not using all persons!')

    results = load('results_valence')
    if results == None:
        print('[warn] rebuilding cache')
        pool = Pool(processes=POOL_SIZE)
        results = pool.map(getPersonRankings, range(1,stop_person))
        pool.close()
        pool.join()
        dump(results,'results_valence')

    results = np.array(results)
    genReport(results)

    accs = load("accs")
    if accs == None:
        print('[warn] rebuilding cache')
        accs = top30Accs(results)
        dump(accs,'accs')
    acc = np.array(accs)
    accReport(accs)
