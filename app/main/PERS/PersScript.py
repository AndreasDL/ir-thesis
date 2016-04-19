from sklearn import svm
from scipy.stats import pearsonr, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mutual_info_score
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_regression, SelectKBest

import personLoader
import Classificators
import featureExtractor as FE
from personLoader import load, dump
import numpy as np

from multiprocessing import Pool
POOL_SIZE = 3


class PersScript():
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

        self.ddpad = "../../../dumpedData/persScript/"
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

    def d_n(self, x):
        d = np.abs(x[:, None] - x)
        dn = d - d.mean(0) - d.mean(1)[:, None] + d.mean()
        return dn
    def dcov_all(self, x, y):
        # stolen with pride from https://gist.githubusercontent.com/josef-pkt/2938402/raw/ff949d6379484e9fc745bd534bdfc5d22ad074e2/try_distance_corr.py
        dnx = self.d_n(x)
        dny = self.d_n(y)

        denom = np.product(dnx.shape)
        dc = (dnx * dny).sum() / denom
        dvx = (dnx ** 2).sum() / denom
        dvy = (dny ** 2).sum() / denom
        dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
        return np.sqrt(dc), np.sqrt(dr), np.sqrt(dvx), np.sqrt(dvy)
    def accuracy(self, predictions, truths):
        acc = 0
        for pred, truth in zip(predictions, truths):
            acc += (pred == truth)

        return acc / float(len(predictions))
    def runPers(self,person):
        #load person data
        pers_results = load('pers_res_p' + str(person), path=self.ddpad)
        if pers_results == None:
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

            #pearson
            corr = []
            for index in range(len(X[0])):
                corr.append( pearsonr(X[:, index], y_cont)[0] )
            pers_results.append(corr)

            #Mut inf
            #dcorr
            mi = []
            dcorr = []
            for feature in np.transpose(X):
                # normalized mutual information
                c_xy = np.histogram2d(feature, y_cont, 2)[0]
                entX = entropy(feature, y_cont)
                entY = entropy(feature, y_cont)
                nMutInf = mutual_info_score(None, None, contingency=c_xy) / float(np.sqrt(entX * entY))
                mi.append(nMutInf)

                # Distance Correlation
                dc, dr, dvx, dvy = self.dcov_all(feature, y_cont)
                dcorr.append(dr)

            pers_results.append(mi)
            pers_results.append(dcorr)

            #Linear Regression
            lr = LinearRegression(n_jobs=-1)
            lr.fit(X, y_cont)
            pers_results.append(lr.coef_)

            #Lasso Regression
            alphas = [0.03, 0.1, 0.3, 1, 3, 10]
            best_alpha = 0.01
            best_acc = 0
            for train_index, test_index in KFold(len(y_cont), n_folds=5):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_cont[train_index], y_cont[test_index]

                lasso = Lasso(alpha=best_alpha)
                lasso.fit(X_train, y_train)
                pred = lasso.predict(X_test)
                best_acc += self.accuracy(pred, y_cont)
            best_acc /= float(5)

            for alpha in alphas:
                acc = 0
                for train_index, test_index in KFold(len(y_cont), n_folds=5):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y_cont[train_index], y_cont[test_index]

                    lasso = Lasso(alpha=alpha)
                    lasso.fit(X_train, y_train)
                    pred = lasso.predict(X_test)
                    acc += self.accuracy(pred, y_test)

                acc /= float(5)
                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha

            lasso = Lasso(alpha=best_alpha)
            lasso.fit(X, y_cont)
            pers_results.append(lasso.coef_)

            #Ridge Regression
            alphas = [0.03, 0.1, 0.3, 1, 3, 10]
            best_alpha = 0.01
            best_acc = 0
            for train_index, test_index in KFold(len(y_cont), n_folds=5):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_cont[train_index], y_cont[test_index]

                ridge = Ridge(alpha=best_alpha)
                ridge.fit(X_train, y_train)
                pred = ridge.predict(X_test)
                best_acc += self.accuracy(pred, y_test)
            best_acc /= float(5)

            for alpha in alphas:
                acc = 0
                for train_index, test_index in KFold(len(y_cont), n_folds=5):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y_cont[train_index], y_cont[test_index]

                    ridge = Ridge(alpha=alpha)
                    ridge.fit(X_train, y_train)
                    pred = lasso.predict(X_test)
                    acc += self.accuracy(pred, y_test)

                acc /= float(5)
                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha

            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X, y_cont)
            pers_results.append(ridge.coef_)

            #SVM
            clf = svm.SVC(kernel='linear')
            clf.fit(X, y_disc)
            svm_weights = (clf.coef_ ** 2).sum(axis=0)
            svm_weights /= float(svm_weights.max())
            pers_results.append(svm_weights)

            #Random Forests
            #rf importances
            #grow forest
            forest = RandomForestClassifier(
                n_estimators=2000,
                max_features='auto',
                criterion='gini',
                n_jobs=-1,
            )
            forest.fit(X,y_disc)
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
            pers_results.append(importances)
            pers_results.append(std)

            #ANOVA
            anova = SelectKBest(f_regression, k=self.threshold)
            anova.fit(X,y_disc)
            selected_features = anova.get_support()
            pers_results.append(selected_features)

            #Linear Discriminant Analysis
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(X,y_disc)
            pers_results.append(lda.coef_[0])

            #Principal Component Analysis
            pca = PCA(n_components=1)
            pca.fit(X)
            pers_results.append(pca.components_[0])

            dump(pers_results, 'pers_res_p' + str(person), path=self.ddpad)

        return np.array(pers_results)

    def run(self):

        pool = Pool(processes=POOL_SIZE)
        rankings = pool.map(self.runPers, range(1, self.stopperson + 1))
        pool.close()
        pool.join()

        print(rankings)

if __name__ == '__main__':
    PersScript("PHY",2,20,Classificators.ContArousalClassificator()).run()