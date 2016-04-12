import personLoader
from personLoader import load,dump
import Classificators
import featureExtractor as FE

from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import numpy as np
import datetime
import time
from multiprocessing import Pool
POOL_SIZE = 2

NESTIMATORS = 100
STOPPERSON = 32  # load this many persons
FOLDS = 5  # fold for out of sample acc
TOPFEATCOUNT = 20  # take 1 -> this amount of features for graph


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

#distance cov
def d_n(x):
    d = np.abs(x[:, None] - x)
    dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean()
    return dn
def dcov_all(x, y):
    #stolen with pride from https://gist.githubusercontent.com/josef-pkt/2938402/raw/ff949d6379484e9fc745bd534bdfc5d22ad074e2/try_distance_corr.py
    dnx = d_n(x)
    dny = d_n(y)

    denom = np.product(dnx.shape)
    dc = (dnx * dny).sum() / denom
    dvx = (dnx**2).sum() / denom
    dvy = (dny**2).sum() / denom
    dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
    return np.sqrt(dc), np.sqrt(dr), np.sqrt(dvx), np.sqrt(dvy)

def getPersonRankings(person):
    pers_results = []

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

    y_disc = np.array(y_cont)
    y_disc[ y_disc <= 5 ] = 0
    y_disc[ y_disc >  5 ] = 1

    #manual Feature standardization
    X = X - np.average(X,axis=0)
    X = np.true_divide(X, np.std(X,axis=0) )

    #statistical tests
    #get pearson
    corr = []
    for index in range(len(X[0])):
        corr.append( pearsonr(X[:, index], y_cont)[0] )
    pers_results.append(corr)

    # mutual information
    mi = []
    for feature in np.transpose(X):
        c_xy = np.histogram2d(feature, y_cont, 2)[0]
        mi.append( mutual_info_score(None, None, contingency=c_xy) )
    pers_results.append(mi)

    dcorr = []
    for feature in np.transpose(X):
        dc, dr, dvx, dvy = dcov_all(feature, y_cont)
        dcorr .append(dr)
    pers_results.append(dcorr)

    #model based:
    #normal regression
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X, y_cont)
    pers_results.append(lr.coef_)

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
    pers_results.append(lasso.coef_)

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
    pers_results.append(ridge.coef_)

    #svm coefficients
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y_disc)
    svm_weights = (clf.coef_ ** 2).sum(axis=0)
    svm_weights /= float(svm_weights.max())
    pers_results.append(svm_weights)

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
    pers_results.append(importances)

    # pca coef
    pca = PCA(n_components=1)
    pca.fit(X)
    pers_results.append(pca.components_[0])

    return np.array(pers_results)
def genReport(results):
    what_mapper = {
        FE.PSDExtractor: 'PSD',
        FE.DEExtractor: 'DE',
        FE.RASMExtractor: 'RASM',
        FE.DASMExtractor: 'DASM',
        FE.AlphaBetaExtractor: 'AB',
        FE.DCAUExtractor: 'DCAU',
        FE.RCAUExtractor: 'RCAU',
        FE.LMinRLPlusRExtractor: 'LminR',
        FE.FrontalMidlinePower: 'FM',
        FE.AvgExtractor: 'AVG',
        FE.STDExtractor: 'STD',
        FE.AVGHeartRateExtractor: 'AVG HR',
        FE.STDInterBeatExtractor: 'STD HR'
    }

    #take averages
    # results[person][metric][feature]
    avg_results = [x[:] for x in [[0] * len(results[0][0])] * len(results[0])]
    for person in range(len(results)): #foreach person
        for metric in range(len(results[person])): #foreach metric
            for feature in range(len(results[person][metric])):
                avg_results[metric][feature] += results[person][metric][feature]
    avg_results = np.array(avg_results)
    avg_results = np.true_divide(avg_results,float(len(results))) #divide by person count

    avg_results = np.transpose(avg_results) #transpose for write to report

    #output to file
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
    f = open('../../results/ranking_valence' + str(st) + ".csv", 'w')

    f.write('featname;eeg/phy;what;channels;waveband;' +
            'pearson_r;mi;dcorr;' +
            'lr coef;l1 coef;l2 coef;' +
            'svm coef;rf importances;PCA\n')
    for featExtr, result in zip(getFeatures().featureExtrs, avg_results):

        feat_name = featExtr.featureName
        feat_what = what_mapper[type(featExtr)]
        feat_eeg, feat_channel, feat_waveband = None, None, None

        if type(featExtr) in [FE.AVGHeartRateExtractor, FE.AvgExtractor, FE.STDExtractor, FE.STDInterBeatExtractor]:
            feat_eeg = 'phy'
            feat_waveband = 'n/a' #no freq bands
            feat_channel = 'n/a'
        else:
            feat_eeg = 'eeg'

            #single channel
            if type(featExtr) in [FE.PSDExtractor, FE.DEExtractor, FE.FrontalMidlinePower]:
                feat_waveband = featExtr.usedFeqBand
                feat_channel = FE.all_channels[featExtr.usedChannelIndexes[0]]

            elif type(featExtr) == FE.AlphaBetaExtractor:
                feat_waveband = 'alpha & beta'
                feat_channel = FE.all_channels[featExtr.usedChannelIndexes[0]]

            elif type(featExtr) == FE.LMinRLPlusRExtractor:
                feat_waveband = 'alpha'
                feat_channel = [
                    FE.all_channels[featExtr.left_channels[0]],
                    FE.all_channels[featExtr.right_channels[0]]
                ]

            #multiple channels Left and right
            elif type(featExtr) in [FE.DASMExtractor, FE.RASMExtractor]:
                feat_waveband = featExtr.leftDEExtractor.usedFeqBand
                feat_channel = [
                    FE.all_channels[featExtr.leftDEExtractor.usedChannelIndexes[0]],
                    FE.all_channels[featExtr.rightDEExtractor.usedChannelIndexes[0]]
                ]

            #multiple channels post and front
            else:
                feat_waveband = featExtr.frontalDEExtractor.usedFeqBand
                feat_channel = [
                    FE.all_channels[featExtr.frontalDEExtractor.usedChannelIndexes[0]],
                    FE.all_channels[featExtr.posteriorDEExtractor.usedChannelIndexes[0]]
                ]

        f.write(
            str(feat_name) + ';' +
            str(feat_eeg) + ';' +
            str(feat_what) + ';' +
            str(feat_channel) + ';' +
            str(feat_waveband) + ';'
        )

        for metric in result:
            f.write(str(abs(metric)) + str(';'))
        f.write("\n")

        #result[metric][feature]
        #for feature in range(len(result[0][0])):
        #    for metric in range(len(result[0])):
        #        f.write(str(abs(avg_results[metric][feature])) + ";")
        #    f.write("\n")

    f.close()

def fixStructure(all_X, all_y_disc):
    # structure of X
    X, y_disc = [], []
    for person_x, person_y in zip(all_X, all_y_disc):
        for video, label in zip(person_x, person_y):
            X.append(video)
            y_disc.append(label)

    X = np.array(X)
    y_disc = np.array(y_disc)

    return np.array(X), np.array(y_disc)
def reverseFixStructure(X, y_disc):
    persons_x, persons_y = [], []
    person_x, person_y = [], []
    for index, (video, label) in enumerate(zip(X, y_disc)):

        if index % 40 == 0 and index != 0: #start of a new person
            persons_x.append(person_x)
            persons_y.append(person_y)

            person_x = []
            person_y = []

        person_x.append(video)
        person_y.append(label)

    persons_x.append(person_x)
    persons_y.append(person_y)


    return np.array(persons_x), np.array(persons_y)

def getAccs(results):
    # load persons
    # load list[person][video][feature] = val
    featExtrs = getFeatures()
    featNames = np.array(featExtrs.getFeatureNames())

    all_y_disc = load('all_pers_y')
    if all_y_disc == None:
        print("[warn] rebuilding cache")
        all_X, all_y_disc = personLoader.PersonsLoader(
            classificator=Classificators.ContValenceClassificator,  # will return disc values anyway!
            featExtractor=featExtrs,
            stopPerson=STOPPERSON
        ).load()
        dump(all_y_disc, 'all_pers_y')
        dump(all_X, 'all_pers_X')
    else:
        all_X = load('all_pers_X')

    all_X = np.array(all_X)
    all_y_disc = np.array(all_y_disc)

    # train test split #TODO

    X, y_disc = fixStructure(all_X, all_y_disc)

    # manual Feature standardization
    X = X - np.average(X, axis=0)
    X = np.true_divide(X, np.std(X, axis=0))

    # take averages
    # results[person][metric][feature]
    avg_results = [x[:] for x in [[0] * len(results[0][0])] * len(results[0])]
    for person in range(len(results)):  # foreach person
        for metric in range(len(results[person])):  # foreach metric
            for feature in range(len(results[person][metric])):
                avg_results[metric][feature] += results[person][metric][feature]
    avg_results = np.array(avg_results)
    avg_results = np.true_divide(avg_results, float(len(results)))  # divide by person count
    # avg_results[metric][feature]

    metric_results = []
    for metric in range(len(avg_results)):
        # sort features
        indices = np.array(np.argsort(avg_results[metric])[::-1])

        # get first TOPFEATCOUNT
        #featurenames
        featNames_metric = np.array(indices[:TOPFEATCOUNT])

        feat_results = []
        for featCount in range(1, TOPFEATCOUNT + 1):
            print('metric: ' + str(metric) + ' - featCount: ' + str(featCount))

            # filter features out X
            X_filtered = X[:, indices[:featCount]]

            #old struct
            X_old, y_old = reverseFixStructure(X_filtered,y_disc)

            # 5 fold
            train_acc, test_acc = 0, 0
            for train_index, test_index in KFold(len(y_old), n_folds=FOLDS, random_state=19, shuffle=True):
                #new struct
                X_train, y_train = fixStructure(X_old[train_index], y_old[train_index])
                X_test , y_test  = fixStructure(X_old[test_index ], y_old[test_index ])

                clf = RandomForestClassifier(
                    n_estimators=NESTIMATORS,
                    max_features='auto',
                    criterion='gini',
                    n_jobs=-1,
                )
                clf.fit(X_train, y_train)

                train_acc += accuracy(clf.predict(X_train), y_train)
                test_acc += accuracy(clf.predict(X_test), y_test)

            feat_results.append([
                featNames[featNames_metric[featCount - 1]],
                train_acc / float(FOLDS),
                test_acc  / float(FOLDS)
            ])

        metric_results.append(feat_results) #metricresults[featCount] = [featname, train, test]

    return metric_results
def genAccReport(accs):
    #accs[metric][featCount -1] = (name,train,test)

    # output to file
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
    f = open('../../results/accs_rf_valence' + str(st) + ".csv", 'w')

    for index, metric in enumerate(accs):
        f.write("Metric: " + str(index + 1) + ";\nfeatureCount;featureAdded;train;test;\n")
        for featCount, line in enumerate(metric):
            f.write(str(featCount+1) + ";")
            for column in line:
                f.write(str(column) + ";")
            f.write("\n")

        f.write("\n")
        f.write("\n")

    f.close()

if __name__ == '__main__':


    if STOPPERSON < 32:
        print('[warn] not using all persons')

    if FOLDS > STOPPERSON:
        print('[warn] folds > stopperson --quitting')
        exit

    results = load('results_valence')
    if results == None:
        print('[warn] rebuilding valence results cache')
        pool = Pool(processes=POOL_SIZE)
        results = pool.map(getPersonRankings, range(1, STOPPERSON+1))
        pool.close()
        pool.join()
        dump(results, 'results_valence')

    results = np.array(results)
    #results[person][feature][metric]
    genReport(results)

    accs = load('accs')
    if accs == None:
        print("[warn] rebuilding accs cache")
        accs = getAccs(results)
        dump(accs, 'accs')

    accs = np.array(accs)
    genAccReport(accs)
