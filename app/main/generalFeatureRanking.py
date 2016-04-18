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
POOL_SIZE = 3

class persScript():

    def __init__(self, feats, stopperson, threshold, classifier):
        self.featExtr = FE.MultiFeatureExtractor()
        if feats == 'EEG':
            self.addEEGFeatures()
        elif feats == 'PHY':
            self.addPhyFeatures()
        else:
            self.addEEGFeatures()
            self.addPhyFeatures()

        self.stopperson = stopperson
        self.threshold = threshold

        self.classifier = classifier

        self.ddpad = "../../dumpedData/genScript/"
        if self.classifier.name == "ContArousalClasses":
            self.ddpad += "arousal/"
        else:
            self.ddpad += "valence/"

        if feats == "EEG":
            self.ddpad += "eeg/"
        elif feats == "PHY":
            self.ddpad += "phy/"
        else:
            self.ddpad += "all/"

        self.rpad = self.ddpad + "results/"

    def addEEGFeatures(self):

        # EEG
        for channel in FE.all_EEG_channels:

            self.featExtr.addFE(
                FE.AlphaBetaExtractor(
                    channels=[channel],
                    featName='A/B ' + FE.all_channels[channel]
                )
            )

            for freqband in FE.startFreq:

                if freqband != 'all':
                    self.featExtr.addFE(
                        FE.BandFracExtractor(
                            channels=[channel],
                            featName=str(channel) + "-" + str(freqband),
                            freqBand=freqband
                        )
                    )

                self.featExtr.addFE(
                    FE.DEExtractor(
                        channels=[channel],
                        freqBand=freqband,
                        featName='DE ' + FE.all_channels[channel] + '(' + freqband + ')'
                    )
                )

                self.featExtr.addFE(
                    FE.PSDExtractor(
                        channels=[channel],
                        freqBand=freqband,
                        featName='PSD ' + FE.all_channels[channel] + '(' + freqband + ')'
                    )
                )

        for left, right in zip(FE.all_left_channels, FE.all_right_channels):

            for freqband in FE.startFreq:
                self.featExtr.addFE(
                    FE.DASMExtractor(
                        left_channels=[left],
                        right_channels=[right],
                        freqBand=freqband,
                        featName='DASM ' + FE.all_channels[left] + ',' + FE.all_channels[right]
                    )
                )

                self.featExtr.addFE(
                    FE.RASMExtractor(
                        left_channels=[left],
                        right_channels=[right],
                        freqBand=freqband,
                        featName='RASM ' + FE.all_channels[left] + ',' + FE.all_channels[right]
                    )
                )

        for front, post in zip(FE.all_frontal_channels, FE.all_posterior_channels):
            for freqband in FE.startFreq:
                self.featExtr.addFE(
                    FE.DCAUExtractor(
                        frontal_channels=[front],
                        posterior_channels=[post],
                        freqBand=freqband,
                        featName='DCAU ' + FE.all_channels[front] + ',' + FE.all_channels[post]
                    )
                )

                self.featExtr.addFE(
                    FE.RCAUExtractor(
                        frontal_channels=[front],
                        posterior_channels=[post],
                        freqBand=freqband,
                        featName='RCAU ' + FE.all_channels[front] + ',' + FE.all_channels[post]
                    )
                )
    def addPhyFeatures(self):
        # physiological signals
        for channel in FE.all_phy_channels:
            self.featExtr.addFE(FE.AvgExtractor(channel, ''))
            self.featExtr.addFE(FE.STDExtractor(channel, ''))
            self.featExtr.addFE(FE.MaxExtractor(channel, ''))
            self.featExtr.addFE(FE.MinExtractor(channel, ''))
            self.featExtr.addFE(FE.MedianExtractor(channel, ''))
            self.featExtr.addFE(FE.VarExtractor(channel, ''))

        self.featExtr.addFE(FE.AVGHeartRateExtractor())
        self.featExtr.addFE(FE.STDInterBeatExtractor())
        self.featExtr.addFE(FE.MaxHRExtractor())
        self.featExtr.addFE(FE.MinHRExtractor())
        self.featExtr.addFE(FE.MedianHRExtractor())
        self.featExtr.addFE(FE.VarHRExtractor())

    #distance cov
    def d_n(self,x):
        d = np.abs(x[:, None] - x)
        dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean()
        return dn
    def dcov_all(self,x, y):
        #stolen with pride from https://gist.githubusercontent.com/josef-pkt/2938402/raw/ff949d6379484e9fc745bd534bdfc5d22ad074e2/try_distance_corr.py
        dnx = self.d_n(x)
        dny = self.d_n(y)

        denom = np.product(dnx.shape)
        dc = (dnx * dny).sum() / denom
        dvx = (dnx**2).sum() / denom
        dvy = (dny**2).sum() / denom
        dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
        return np.sqrt(dc), np.sqrt(dr), np.sqrt(dvx), np.sqrt(dvy)

    def getPersonRankings(self,person):
        pers_results = []

        #load all features & keep them in memory
        y_cont = load('cont_y_p' + str(person), path=self.ddpad)
        if y_cont == None:
            print('[Warn] Rebuilding cache -  person ' + str(person))
            personLdr = personLoader.NoTestsetLoader(self.classifier, self.featExtr)

            X, y_cont = personLdr.load(person)

            dump(X,'X_p' + str(person), path=self.ddpad)
            dump(y_cont,'cont_y_p' + str(person), path=self.ddpad)
        else:
            X = load('X_p' +str(person), path=self.ddpad)

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
            dc, dr, dvx, dvy = self.dcov_all(feature, y_cont)
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
            best_acc += self.accuracy(pred,y_cont)
        best_acc /= float(5)

        for alpha in alphas:
            acc = 0
            for train_index, test_index in KFold(len(y_cont), n_folds=5):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_cont[train_index], y_cont[test_index]

                lasso = Lasso(alpha=alpha)
                lasso.fit(X_train,y_train)
                pred = lasso.predict(X_test)
                acc += self.accuracy(pred,y_test)

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
            best_acc += self.accuracy(pred,y_test)
        best_acc /= float(5)

        for alpha in alphas:
            acc = 0
            for train_index, test_index in KFold(len(y_cont), n_folds=5):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_cont[train_index], y_cont[test_index]

                ridge = Ridge(alpha=alpha)
                ridge.fit(X_train,y_train)
                pred = lasso.predict(X_test)
                acc += self.accuracy(pred,y_test)

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

    def accuracy(self, predictions, truths):
        acc = 0
        for pred, truth in zip(predictions, truths):
            acc += (pred == truth)

        return acc / float(len(predictions))
    def genReport(self):
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
            FE.MaxExtractor: 'MAX',
            FE.MinExtractor: 'Min',
            FE.MedianExtractor: 'Median',
            FE.VarExtractor: 'Var',
            FE.AVGHeartRateExtractor: 'AVG HR',
            FE.STDInterBeatExtractor: 'STD HR',
            FE.MinHRExtractor: 'Min HR',
            FE.MaxHRExtractor: 'Max HR',
            FE.MedianHRExtractor: 'Med HR',
            FE.VarHRExtractor: 'Var HR'
        }

        #take averages
        # results[person][metric][feature]
        avg_results = [x[:] for x in [[0] * len(self.results[0][0])] * len(self.results[0])]
        for person in range(len(self.results)): #foreach person
            for metric in range(len(self.results[person])): #foreach metric
                for feature in range(len(self.results[person][metric])):
                    avg_results[metric][feature] += self.results[person][metric][feature]
        avg_results = np.array(avg_results)
        avg_results = np.true_divide(avg_results,float(len(self.results))) #divide by person count

        avg_results = np.transpose(avg_results) #transpose for write to report

        #output to file
        f = open(self.rpad + 'genScript.csv', 'w')

        f.write('featname;eeg/phy;what;channels;waveband;' +
                'pearson_r;mi;dcorr;' +
                'lr coef;l1 coef;l2 coef;' +
                'svm coef;rf importances;PCA\n')
        for featExtr, result in zip(self.featExtr.featureExtrs, avg_results):

            feat_name = featExtr.featureName
            feat_what = what_mapper[type(featExtr)]
            feat_eeg, feat_channel, feat_waveband = None, None, None

            if type(featExtr) in [FE.AVGHeartRateExtractor, FE.AvgExtractor  , FE.STDExtractor,
                                  FE.STDInterBeatExtractor, FE.MinExtractor  , FE.MaxExtractor,
                                  FE.MedianExtractor      , FE.MinHRExtractor, FE.MaxHRExtractor,
                                  FE.MedianHRExtractor    , FE.VarExtractor  , FE.VarHRExtractor]:
                feat_eeg = 'phy'
                feat_waveband = 'n/a' #no freq bands
                feat_channel = 'n/a'
            else:
                feat_eeg = 'eeg'

                #single channel
                if type(featExtr) in [FE.PSDExtractor, FE.DEExtractor, FE.FrontalMidlinePower, FE.MinExtractor, FE.MaxExtractor, FE.MedianExtractor]:
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

    def getAccs(self,person):
        # load persons
        # load list[person][video][feature] = val
        featNames = np.array(self.featExtr.getFeatureNames())

        pers_results = []

        # load all features & keep them in memory
        y_cont = load('cont_y_p' + str(person), path=self.ddpad)
        if y_cont == None:
            print('[Warn] Rebuilding cache -  person ' + str(person))
            personLdr = personLoader.NoTestsetLoader(self.classifier, self.featExtr)

            X, y_cont = personLdr.load(person)

            dump(X, 'X_p' + str(person), path=self.ddpad)
            dump(y_cont, 'cont_y_p' + str(person), path=self.ddpad)
        else:
            X = load('X_p' + str(person), path=self.ddpad)

        y_disc = np.array(y_cont)
        y_disc[y_disc <= 5] = 0
        y_disc[y_disc > 5] = 1

        # manual Feature standardization
        X = X - np.average(X, axis=0)
        X = np.true_divide(X, np.std(X, axis=0))

        #take top feat

        #add feature & check if acc is better

        #report acc

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
            featNames_metric = np.array(indices[:self.threshold])

            feat_results = []
            for featCount in range(1, self.threshold + 1):
                print('metric: ' + str(metric) + ' - featCount: ' + str(featCount))

                # filter features out X
                X_filtered = X[:, indices[:featCount]]

                #old struct
                X_old, y_old = self.reverseFixStructure(X_filtered,y_disc)

                # 5 fold
                train_acc, test_acc = 0, 0
                for train_index, test_index in KFold(len(y_old), n_folds=5, random_state=19, shuffle=True):
                    #new struct
                    X_train, y_train = self.fixStructure(X_old[train_index], y_old[train_index])
                    X_test , y_test  = self.fixStructure(X_old[test_index ], y_old[test_index ])

                    clf = RandomForestClassifier(
                        n_estimators=1000,
                        max_features='auto',
                        criterion='gini',
                        n_jobs=-1,
                    )
                    clf.fit(X_train, y_train)

                    train_acc += self.accuracy(clf.predict(X_train), y_train)
                    test_acc += self.accuracy(clf.predict(X_test), y_test)

                feat_results.append([
                    featNames[featNames_metric[featCount - 1]],
                    train_acc / float(5),
                    test_acc  / float(5)
                ])

            metric_results.append(feat_results) #metricresults[featCount] = [featname, train, test]

        return metric_results
    def genAccReport(self,accs):
        #accs[metric][featCount -1] = (name,train,test)

        # output to file
        f = open(self.rpad + 'accs_rf_valence.csv', 'w')

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

    def run(self):

        self.results = load('results_valence', path=self.ddpad)
        if self.results == None:
            print('[warn] rebuilding valence results cache')
            pool = Pool(processes=POOL_SIZE)
            self.results = pool.map(self.getPersonRankings, range(1, self.stopperson + 1))
            pool.close()
            pool.join()
            dump(self.results, 'results_valence', path=self.ddpad)

        self.results = np.array(self.results)
        # results[person][feature][metric]
        self.genReport()

        accs = load('accs', self.ddpad)
        if accs == None:
            print("[warn] rebuilding accs cache")
            accs = self.getAccs()
            dump(accs, 'accs', path=self.ddpad)

        accs = np.array(accs)
        self.genAccReport(accs)

if __name__ == '__main__':
    persScript("PHY",10,20,Classificators.ContArousalClassificator()).run()