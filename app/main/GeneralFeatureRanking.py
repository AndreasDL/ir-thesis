from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import personLoader
import Classificators
import featureExtractor as FE
from scipy.stats import pearsonr
import numpy as np
import datetime
import time
import featureExtractor
from featureExtractor import all_channels

from personLoader import load, dump
from multiprocessing import Pool
POOL_SIZE = 8

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

    # pca coef
    pca = PCA(n_components=1)
    pca.fit(X)
    pca_coef = pca.components_[0]


    # [pearson_r, mutual inf, max inf, dist, l1 coef, l2 coef, svm coef, rf importances, coord search]
    #featExtr = getFeatures()
    #featnames = featExtr.getFeatureNames()
    pers_results = []
    for corr, lr_scores, l1_scores, l2_scores, svm_weights, importances, pca_coef in zip(corr, lr_scores, l1_scores, l2_scores, svm_weights, importances, pca_coef):
        pers_results.append([np.abs(corr[0]), np.abs(lr_scores), np.abs(l1_scores), np.abs(l2_scores), np.abs(svm_weights), np.abs(importances), np.abs(pca_coef)])

    return pers_results#, featExtr.featureExtrs
def ldaRankings(person):
    FOLDS = 5

    # load all features & keep them in memory
    y_cont = load('cont_y_p' + str(person))
    if y_cont == None:
        print('[Warn] Rebuilding cache -  person ' + str(person))
        classificator = Classificators.ContValenceClassificator()
        featExtr = getFeatures()
        personLdr = personLoader.NoTestsetLoader(classificator, featExtr)

        X, y_cont = personLdr.load(person)

        dump(X, 'X_p' + str(person))
        dump(y_cont, 'cont_y_p' + str(person))
    else:
        X = load('X_p' + str(person))

    X = np.array(X)
    y_cont = np.array(y_cont)
    y_disc = y_cont
    y_disc[y_disc <= 5] = 0
    y_disc[y_disc > 5] = 1

    for index, val in enumerate(np.std(X, axis=0)):
        if val == 0:
            print('warning zero std for feature index: ', index, ' (', personLoader.featureExtractor.getFeatureNames()[index])

    # manual Feature standardization
    X = X - np.average(X, axis=0)
    X = np.true_divide(X, np.std(X, axis=0))

    feat_test_error = []
    feat_train_error = []
    X_temp = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] #phy only
    feat_test_error.append(0)
    feat_train_error.append(0)
    for train_index, test_index in KFold(len(y_disc), n_folds=FOLDS, random_state=17, shuffle=True):
        X_train, X_test = X_temp[train_index], X_temp[test_index]
        y_train, y_test = y_disc[train_index], y_disc[test_index]

        clf = LDA(shrinkage='auto', solver='lsqr')
        clf.fit(X_train, y_train)

        feat_train_error[0] += accuracy(clf.predict(X_train), y_train) / float(FOLDS)
        feat_test_error[0] += accuracy(clf.predict(X_test)  , y_test ) / float(FOLDS)

    print("train: " + str(feat_train_error[0]) + " test: " + str(feat_test_error[0]))

    for feat_index in range(10,30):#len(X[0,:])):
        index = feat_index - 9
        X_temp = X[:, [0,1,2,3,4,5,6,7,8,9,feat_index]]

        feat_test_error.append(0)
        feat_train_error.append(0)
        for train_index, test_index in KFold(len(y_disc), n_folds=FOLDS, random_state=17, shuffle=True):
            X_train, X_test = X_temp[train_index], X_temp[test_index]
            y_train, y_test = y_disc[train_index], y_disc[test_index]

            clf = LDA(shrinkage='auto',solver='lsqr')
            clf.fit(X_train, y_train)

            feat_train_error[index] += accuracy(clf.predict(X_train), y_train) / float(FOLDS)
            feat_test_error[index]  += accuracy(clf.predict(X_test) , y_test ) / float(FOLDS)

        print("train: " + str(feat_train_error[index]) + " test: " + str(feat_test_error[index]))

    return [feat_test_error, feat_train_error]

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
            avg_results.append([0,0,0,0,0,0,0])
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
def genLDAReport(train_accs, test_accs):
    # output to file
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
    f = open('../../results/accs_lda_valence' + str(st) + ".csv", 'w')

    # averages
    avg_train_accs = np.average(train_accs, axis=0)
    avg_train_accs = np.transpose(avg_train_accs)

    avg_test_accs = np.average(test_accs, axis=0)
    avg_test_accs = np.transpose(avg_test_accs)

    # write averages
    f.write('number of features;train_err;test_err;\n')
    for featCount, (train_line, test_line) in enumerate(zip(avg_train_accs, avg_test_accs)):
        f.write(str(featCount + 10) + ";")

        f.write(str(train_line) + ';')
        f.write(str(test_line)  + ';')

        f.write("\n")

    f.write("\n")
    f.write("\n")

    # write best accs per person
    max_train_accs = np.amax(train_accs, axis=1)
    max_test_accs  = np.amax(test_accs , axis=1)
    f.write("best accs obtained for each person\n")
    f.write('person;train_err;test_err;\n')
    for person, (train_line, test_line) in enumerate(zip(max_train_accs, max_test_accs)):
        f.write(str(person + 1) + ";")

        f.write(str(train_line) + ';')
        f.write(str(test_line)  + ';')
        f.write("\n")

    f.close()
def accReport(train_accs, test_accs):
    #output to file
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
    f = open('../../results/accs_rf_valence' + str(st) + ".csv", 'w')

    #averages
    avg_train_accs = np.average(train_accs, axis=0)
    avg_train_accs = np.transpose(avg_train_accs)

    avg_test_accs = np.average(test_accs, axis=0)
    avg_test_accs = np.transpose(avg_test_accs)

    #write averages
    f.write('number of features;train_pearson;train_lr;train_l1;train_l2;train_svm;train_rf;train_pca;')
    f.write('test_pearson;test_lr;test_l1;test_l2;test_svm;test_rf;test_pca\n')
    for person, (train_line, test_line) in enumerate(zip(avg_train_accs, avg_test_accs)):
        f.write(str(person+1) + ";" )

        for acc in train_line:
            f.write(str(acc[0]) + ';')

        for acc in test_line:
            f.write(str(acc[0]) + ';')

        f.write("\n")

    f.write("\n")
    f.write("\n")

    #write best accs per person
    max_train_accs = np.amax(train_accs, axis=3)
    max_test_accs  = np.amax(test_accs , axis=3)
    f.write("best accs obtained for each person")
    f.write('person;train_pearson;train_lr;train_l1;train_l2;train_svm;train_rf;train_pca;')
    f.write('test_pearson;test_lr;test_l1;test_l2;test_svm;test_rf;test_pca;\n')
    for person, (train_line, test_line) in enumerate(zip(max_train_accs,max_test_accs)):
        f.write(str(person+1) + ";" )
        for acc in train_line[0]:
            f.write(str(acc) + ';')

        for acc in test_line[0]:
            f.write(str(acc) + ';')

        f.write("\n")

    f.close()

class accWorker():
    def __init__(self,results, stop_person):
        self.results = results
        self.stop_person = stop_person

    def top30Accs(self,person):
        FOLDS = 5
        # take averages
        avg_results = [[0] * len(self.results[0, 0])] * len(self.results[0])
        for result in self.results:
            for i in range(len(result)):  # each feature
                for j in range(len(result[i])):
                    avg_results[i][j] += result[i][j]
            avg_results = np.array(avg_results)
            avg_results = np.true_divide(avg_results, float(len(self.results)))

        acc_train_no_features = []
        acc_test_no_features = []
        #for person in range(1, len(results) + 1):
        print('working on person ' + str(person))
        # load all features & keep them in memory
        y_cont = load('cont_y_p' + str(person))
        X = load('X_p' + str(person))

        X = np.array(X)
        y_cont = np.array(y_cont)
        y_disc = y_cont
        y_disc[y_disc <= 5] = 0
        y_disc[y_disc > 5] = 1

        for index, val in enumerate(np.std(X, axis=0)):
            if val == 0:
                print('warning zero std for feature index: ', index, ' (', personLoader.featureExtractor.getFeatureNames()[index])

        # manual Feature standardization
        X = X - np.average(X, axis=0)
        X = np.true_divide(X, np.std(X, axis=0))

        pers_train_accs = []
        pers_test_accs = []
        for metric in range(len(avg_results[0])): #for each metric
            # sort features
            indices = np.array(np.argsort(avg_results[:, metric])[::-1])

            # get first 30 indexes
            features_to_keep = indices[:30]
            X_filtered = X[:, features_to_keep]

            model_train_accs = []
            model_test_accs = []
            for i in range(1, len(features_to_keep) + 1): #for each feat count
                print('[' + str(person) + '] - metric: ' + str(metric) + '-' + str(i))
                X_temp = X_filtered[:, :i]

                train_acc, test_acc = 0, 0
                for train_index, test_index in KFold(len(y_disc), n_folds=FOLDS, random_state=19, shuffle=True):
                    X_train, X_test = X_temp[train_index], X_temp[test_index]
                    y_train, y_test = y_disc[train_index], y_disc[test_index]

                    '''
                    clf = RandomForestClassifier(
                        n_estimators=2000,
                        max_features='auto',
                        criterion='gini',
                        n_jobs=-1,
                    )
                    clf.fit(X_train, y_train)
                    '''
                    clf = svm.SVC()
                    clf.fit(X_train, y_train)

                    train_acc += accuracy(clf.predict(X_train), y_train) / float(FOLDS)
                    test_acc  += accuracy(clf.predict(X_test) , y_test ) / float(FOLDS)

                model_train_accs.append(train_acc)
                model_test_accs.append(test_acc)

            pers_train_accs.append(model_train_accs)
            pers_test_accs.append(model_test_accs)

        acc_train_no_features.append(pers_train_accs)
        acc_test_no_features.append(pers_test_accs)

        return [acc_train_no_features, acc_test_no_features]

    def getAccs(self):
        pool = Pool(processes=POOL_SIZE)
        results = pool.map(self.top30Accs, range(1,self.stop_person))
        pool.close()
        pool.join()

        return results

if __name__ == '__main__':

    stop_person = 33
    if stop_person < 33:
        print('[warn] not using all persons!')

    # lda
    lda_results = load('results_valence_lda')
    if lda_results == None:
        print('[warn] rebuilding valence lda results cache')
        pool = Pool(processes=POOL_SIZE)
        lda_results = pool.map(ldaRankings, range(1, stop_person))
        pool.close()
        pool.join()
        dump(lda_results, 'results_valence_lda')

    lda_results = np.array(lda_results)
    train_accs = np.array(lda_results[:, 0])
    test_accs = np.array(lda_results[:, 1])
    genLDAReport(train_accs, test_accs)

    results = load('results_valence')
    if results == None:
        print('[warn] rebuilding valence results cache')
        pool = Pool(processes=POOL_SIZE)
        results = pool.map(getPersonRankings, range(1, stop_person))
#        results = pool.map(ldaRankings, range(1, stop_person))
        pool.close()
        pool.join()
        dump(results,'results_valence')

    results = np.array(results)
    genReport(results)

    acc_results = load('acc_results')
    if acc_results == None:
        print('[warn] rebuilding acc cache')
        acc_results = accWorker(results,stop_person).getAccs()
        dump(acc_results,'acc_results')

    acc_results = np.array(acc_results)
    train_accs = np.array(acc_results[:,0])
    test_accs  = np.array(acc_results[:,1] )

    accReport(train_accs, test_accs)