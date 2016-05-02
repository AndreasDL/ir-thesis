from sklearn.ensemble import RandomForestClassifier

import personLoader
from personLoader import load, dump
import featureExtractor as FE
import Classificators

from PersTree import PersTree
from sklearn.cross_validation import KFold, train_test_split

import numpy as np
import datetime
import time
import matplotlib.pyplot as plt



class RFGen():
    def __init__(self, feats, stopperson, threshold, classifier, runs=40, n_estimators=1000):
        self.featExtr = FE.MultiFeatureExtractor()
        if feats == 'EEG':
            self.addEEGFeatures()
        elif feats == 'PHY':
            self.addPhyFeatures()
        else:
            self.addEEGFeatures()
            self.addPhyFeatures()

        self.stopperson = stopperson
        self.runs = runs
        self.n_estimators = n_estimators
        self.threshold = threshold

        self.classifier = classifier

        self.ddpad = "../../dumpedData/GenRF/"
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

    def genPlot(self,avgs, stds, title):

        #st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M%S')
        fname = self.rpad + str(title) + '.png'

        # Plot the feature importances of the forest
        plt.figure()
        plt.title(title)
        plt.bar(
            range(len(avgs)),
            avgs,
            color="r",
            yerr=stds,
            align="center"
        )
        plt.xticks(range(0, len(avgs), 50))
        plt.xlim([-1, len(avgs)])
        plt.savefig(fname)
        plt.clf()
        plt.close()
    def genDuoPlot(self,avgs1, stds1, avgs2,stds2, title):

        fname = self.rpad + str(title) + '.png'

        # Plot the feature importances of the forest
        fig, ax = plt.subplots()
        N = len(avgs1)
        ind = np.arange(N)
        width = 0.35

        plt.title(title)
        inter = ax.bar(
            ind,
            avgs1,
            color="r",
            yerr=stds1,
            align="center",
            width=width
        )

        pred = ax.bar(
            ind+width,
            avgs2,
            color="b",
            yerr=stds2,
            align="center",
            width=width
        )
        plt.xticks(range(0, len(avgs1), 5))
        plt.xlim([-1, len(avgs1)])
        ax.legend((pred,inter), ('inter', 'pred'))
        plt.savefig(fname)

        #plt.show()
        plt.clf()
        plt.close()

    def step1(self,X,y, featureNames, threshold, criterion='gini'):
        print('step1')

        #get importances
        forest = PersTree(
            n_trees=self.n_estimators
        )

        forest.fit(X,y)

        # get importances
        importances, stds = forest.getImportance()
        importances -= stds

        # genPlot(importances, stds, 'step1 importances'))
        # sort features
        indices_to_keep = np.array(np.argsort(importances)[::-1])
        featureNames = np.array(featureNames)[indices_to_keep]

        #keep upper %threshold
        indices_to_keep = indices_to_keep[:int(len(indices_to_keep)*threshold)]

        #filter indices
        importances = importances[indices_to_keep]
        featureNames = featureNames[indices_to_keep]

        # sort features
        indices_to_keep = np.array(np.argsort(importances)[::-1])
        featureNames = featureNames[indices_to_keep]

        return np.array(indices_to_keep), featureNames

    def step2_prediction(self,X, y, featureNames, criterion='gini'):
        featuresLeft = len(X[0])
        print('step2_prediction - featLeft: ' + str(featuresLeft))

        #for featCount = 1 ~> remaining indices
        best_features_to_keep = []
        best_score, best_std = 0, 0
        for feat in range(featuresLeft):
            print('pred - ' + str(feat))

            forest =PersTree(
                n_trees=self.n_estimators
            )

            # add new feature to the existing features
            features_to_keep = best_features_to_keep[:]
            features_to_keep.append(feat)

            #prepare the dataset
            X_temp = X[:,features_to_keep]

            #get scores
            run_scores = []
            for i in range(self.runs):
                forest.fit(X_temp,y)
                avg, std = forest.getOob()
                run_scores.append(avg)

            new_score = np.average(run_scores)
            new_std   = np.std(run_scores)

            #better?
            if new_score - new_std > best_score - best_std:
                best_score = new_score
                best_std   = new_std
                best_features_to_keep =features_to_keep

        return best_features_to_keep, best_score, best_std

    def genReport(self,result):
        #results[person] = [
        #   [ featCount_inter  , score_inter, std_inter, featureNames_inter, avgs, stds ],
        #   [ len(indices_pred), score_pred , std_pred , featureNames_pred  ]
        #]

        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
        f = open(self.rpad + 'rfgen' + str(st) + ".csv", 'w')
        f.write("interScore;interStd;interCount;interFeat;\n")

        scores = []
        stds   = []
        for i in range(1):
            methodScores = []
            methodSTDs   = []
            f.write(str(result[i][1]) + ';' + str(result[i][2]) + ';' + str(result[i][0]) + ';')
            for name in result[i][3]:
                f.write(str(name) + ';')

            f.write('\n')
            self.genPlot(result[0][4], result[0][5], 'method' + str(i) + ' oob score ')

            methodScores.append(result[i][1])
            methodSTDs.append(result[i][2])

            self.genPlot(methodScores, methodSTDs, 'method' + str(i) + 'scores')
            scores.append(methodScores)
            stds.append(methodSTDs)

            f.write('\n')
            f.write('\n')

            f.write("predScore;predStd;predCount;predFeat;\n")
        f.close()

        self.genDuoPlot(scores[0], stds[0], scores[1], stds[1], 'interpretation vs perdiction scores')

    def fixStructure(self,all_X, all_y_disc):
        # structure of X
        X, y_disc = [], []
        for person_x, person_y in zip(all_X, all_y_disc):
            for video, label in zip(person_x, person_y):
                X.append(video)
                y_disc.append(label)

        X = np.array(X)
        y_disc = np.array(y_disc)

        return np.array(X), np.array(y_disc)
    def reverseFixStructure(self,X, y_disc):
        persons_x, persons_y = [], []
        person_x, person_y = [], []
        for index, (video, label) in enumerate(zip(X, y_disc)):

            if index % 40 == 0 and index != 0:  # start of a new person
                persons_x.append(person_x)
                persons_y.append(person_y)

                person_x = []
                person_y = []

            person_x.append(video)
            person_y.append(label)

        persons_x.append(person_x)
        persons_y.append(person_y)

        return np.array(persons_x), np.array(persons_y)

    def run(self):

        results = load('results')
        if results == None:

            all_y_disc = load('all_pers_y', self.ddpad)
            if all_y_disc == None:
                print("[warn] rebuilding cache")
                all_X, all_y_disc = personLoader.PersonsLoader(
                    classificator=Classificators.ContValenceClassificator(),
                    featExtractor=self.featExtr,
                    stopPerson = self.stopperson
                ).load()
                dump(all_y_disc, 'all_pers_y', self.ddpad)
                dump(all_X, 'all_pers_X', self.ddpad)
            else:
                all_X = load('all_pers_X', self.ddpad)

            all_X = np.array(all_X)
            all_y_disc = np.array(all_y_disc)

            X, y_disc = self.fixStructure(all_X, all_y_disc)

            # manual Feature standardization
            X = X - np.average(X, axis=0)
            X = np.true_divide(X, np.std(X, axis=0))

            # Train / testset
            X, X_test, y_disc, y_test = train_test_split(X,y_disc,test_size=10, random_state=17)

            # back to orig struct
            X, y_disc = self.reverseFixStructure(X,y_disc)

            # step 1 determine importances using RF forest
            temp = load("step1",path=self.ddpad)
            indices_step1, featureNames_step1 = None, None
            if temp == None:
                indices_step1, featureNames_step1 = self.step1(X, y_disc, self.featExtr.getFeatureNames(), self.threshold)
                dump({'indicices': indices_step1,
                      'featNames': featureNames_step1}, "step1", self.ddpad)
            else:
                indices_step1 = temp['indicices']
                featureNames_step1 = temp['featNames']

            featureNames = np.array(featureNames_step1)
            indices = np.array(indices_step1)

            # filter features (X) based on the results from step 1
            X, y_disc = self.fixStructure(X, y_disc)
            X = X[:, indices]
            X, y_disc = self.reverseFixStructure(X, y_disc)

            # step 2 - prediction
            indices_pred, score_pred, std_pred = self.step2_prediction(X, y_disc, featureNames)
            featureNames_pred = featureNames[indices_pred]

            forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_features='auto',
                criterion='gini',
                n_jobs=-1,
                oob_score=True,
                bootstrap=True
            )
            test_accs = []
            test_probs = []

            for i in range(self.runs):
                forest.fit(X,y_disc)
                preds = forest.predict(X_test)
                test_accs.append(self.accuracy(preds),y_test)
                test_probs.append(forest.predict_proba(X_test), y_test)

            test_accs = np.array(test_accs)
            test_probs = np.array(test_probs)
            preds = np.array(preds)

            results = [
                [len(indices_pred), score_pred, std_pred, featureNames_pred, test_accs, np.average(test_accs), np.std(test_accs), y_test, preds ]
            ]

            print('  prediction - score: ' + str(results[0][1]) + ' (' + str(results[0][2]) + ') with ' + str(results[0][0]))

            dump(results, 'results', self.ddpad)

        self.genReport(results)

if __name__ == "__main__":
    RFGen("ALL", 32, 30, Classificators.ContValenceClassificator()).run()
    RFGen("EEG", 32, 30, Classificators.ContValenceClassificator()).run()
    RFGen("PHY", 32, 30, Classificators.ContValenceClassificator()).run()

    RFGen("ALL", 32, 30, Classificators.ContArousalClassificator()).run()
    RFGen("EEG", 32, 30, Classificators.ContArousalClassificator()).run()
    RFGen("PHY", 32, 30, Classificators.ContArousalClassificator()).run()
