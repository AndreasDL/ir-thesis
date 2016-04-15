import personLoader
from personLoader import load, dump
import featureExtractor as FE
import Classificators

from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from multiprocessing import Pool
POOL_SIZE = 2

class SVMPers():

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


        self.ddpad = "../../dumpedData/SVM/"
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

    def step1(self,X,y, featureNames):
        print('step1')

        #get importances

        model = SVC(kernel='linear')

        model.fit(X,y)

        # get importances
        importances = model.coef_

        # genPlot(importances, stds, 'step1 importances'))
        # sort features
        indices_to_keep = np.array(np.argsort(importances)[::-1])
        featureNames = featureNames[indices_to_keep]

        #keep upper %threshold
        indices_to_keep = indices_to_keep[:self.threshold]

        #filter indices
        importances = importances[indices_to_keep]
        featureNames = featureNames[indices_to_keep]

        # sort features
        indices_to_keep = np.array(np.argsort(importances)[::-1])
        featureNames = featureNames[indices_to_keep]

        return importances, np.array(indices_to_keep), featureNames
    def step2_prediction(self,X, y, featureNames):
        featuresLeft = len(X[0])
        print('step2_prediction - featLeft: ' + str(featuresLeft))

        #for featCount = 1 ~> remaining indices
        best_features_to_keep = []
        best_score, best_std = 0, 0
        for feat in range(featuresLeft):
            print('pred - ' + str(feat))

            model = SVC(kernel='linear')

            # add new feature to the existing features
            features_to_keep = best_features_to_keep[:]
            features_to_keep.append(feat)

            #prepare the dataset
            X_temp = X[:,features_to_keep]

            #get scores
            run_scores = []
            for tr, te in KFold(n=len(X_temp), n_folds=5, shuffle=True, random_state=17):

                model.fit(X_temp[tr], y[tr])
                run_scores.append(self.accuracy(model.predict(X_temp[te]), y[te]))

            new_score = np.average(run_scores)
            new_std   = np.std(run_scores)

            #better?
            if new_score - new_std > best_score - best_std:
                best_score = new_score
                best_std   = new_std
                best_features_to_keep =features_to_keep

        return best_features_to_keep, best_score, best_std

    def genReport(self, results):
        #results[person] = [
        #   [ len(indices_pred), score_pred , std_pred , featureNames_pred  , test_acc_avg, test_acc_std]
        #]

        f = open(self.rpad + "results.csv", 'w')
        f.write("person;predScore;predStd;avg_test_acc;std_test_acc;predCount;predFeat;\n")

        methodScores = []
        methodSTDs   = []
        for person,result in enumerate(results):
            f.write(str(person) + ';')
            f.write(str(result[1]) + ';' + str(result[2]) + ';' + str(result[4]) + ';' + str(result[5]) + ';' + str(result[0]) + ';')
            for name in result[3]:
                f.write(str(name) + ';')

            f.write('\n')
            methodScores.append(result[1])
            methodSTDs.append(result[2])
        self.genPlot(methodScores, methodSTDs,'scores')

        f.close()
    def genPlot(self, avgs, stds, title):

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
        plt.xticks(range(len(avgs)))
        plt.xlim([-1, len(avgs)])
        plt.savefig(fname)
        plt.clf()
        plt.close()
    def genDuoPlot(self, avgs1, stds1, avgs2, stds2, title):
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
            ind + width,
            avgs2,
            color="b",
            yerr=stds2,
            align="center",
            width=width
        )
        plt.xticks(range(0, len(avgs1), 5))
        plt.xlim([-1, len(avgs1)])
        ax.legend((inter, pred), ('inter', 'pred'))
        plt.savefig(fname)

        # plt.show()
        plt.clf()
        plt.close()
    def accuracy(self,predictions, truths):
        acc = 0
        for pred, truth in zip(predictions, truths):
            acc += (pred == truth)

        return acc / float(len(predictions))

    def fixPerson(self, person):
        print('person: ' + str(person))

        to_ret = load('P' + str(person), path=self.ddpad)
        if to_ret == None:
            #load X , y
            # load all features & keep them in memory
            featureNames = np.array(self.featExtr.getFeatureNames())

            y_cont = load('cont_y_p' + str(person),path =self.ddpad)
            if y_cont == None:
                print('[Warn] Rebuilding cache -  person ' + str(person))
                X, y_cont = personLoader.NoTestsetLoader(
                    classificator=self.classifier,
                    featExtractor=self.featExtr,
                ).load(person)

                dump(X, 'X_p' + str(person), path=self.ddpad)
                dump(y_cont, 'cont_y_p' + str(person), path=self.ddpad)
            else:
                X = load('X_p' + str(person), path=self.ddpad)

            y_disc = np.array(y_cont)
            y_disc[y_disc <= 5] = 0
            y_disc[y_disc >  5] = 1

            # manual Feature standardization
            X = X - np.average(X, axis=0)
            X = np.true_divide(X, np.std(X, axis=0))

            # Train / testset
            X, X_test, y_disc, y_test = train_test_split(X,y_disc,test_size=10, random_state=17)

            #step 1 determine importances using RF forest
            step1_importances, step1_indices, step1_featureNames = self.step1(X,y_disc, featureNames)
            featureNames = np.array(step1_featureNames)
            indices = np.array(step1_indices)

            #filter features (X) based on the results from step 1
            X = X[:,indices]

            #step 2 - interpretation
            #featCount_inter, score_inter, std_inter, avgs, stds = self.step2_interpretation(X, y_disc, featureNames)
            #indices_inter = indices[:featCount_inter]
            #featureNames_inter = featureNames[indices_inter]

            #step 2 - prediction
            step2_indices_pred, step2_score_pred, step2_std_pred = self.step2_prediction(X, y_disc, featureNames)
            featureNames_pred = featureNames[step2_indices_pred]

            # test err ?
            indices = np.array(step2_indices_pred)
            X = X[:,indices]
            X_test = X_test[:,step2_indices_pred]

            model = SVC(kernel='linear')

            # get scores
            model.fit(X, y_disc)
            test_acc = self.accuracy(model.predict(X_test), y_test)

            to_ret = [
                #[ featCount_inter  , score_inter, std_inter, featureNames_inter, avgs, stds ],
                len(step2_indices_pred), step2_score_pred , step2_std_pred , featureNames_pred, test_acc
            ]

            dump(to_ret, 'P' + str(person), path=self.ddpad)
        print('[' + str(person) + '] prediction - score: ' + str(to_ret[1]) + ' (' + str(to_ret[2]) + ') with ' + str(to_ret[0]) + ' test_score ' + str(to_ret[4]))

            #print('[' + str(person) + '] interpretation - score: ' + str(to_ret[0][1]) + ' (' + str(to_ret[0][2]) + ') with ' + str(to_ret[0][0]) +
        #      '  prediction - score: ' + str(to_ret[1][1]) + ' (' + str(to_ret[1][2]) + ') with ' + str(to_ret[1][0])
        #      )


        return to_ret

    def run(self):

        results = load('data_results', path=self.ddpad)
        if results == None:
            pool = Pool(processes=POOL_SIZE)
            results = pool.map(self.fixPerson, range(1, self.stopperson + 1))
            pool.close()
            pool.join()
            dump(results, 'data_results', path=self.ddpad)

        self.genReport(results)


if __name__ == '__main__':
    SVMPers("ALL", 2, 40, Classificators.ContArousalClassificator()).run()