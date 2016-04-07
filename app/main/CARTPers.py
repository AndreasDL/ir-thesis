import personLoader
from personLoader import load, dump
import featureExtractor as FE
import Classificators

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from pprint import pprint

from multiprocessing import Pool
POOL_SIZE = 2

STOPPERSON = 32
RUNS = 30
N_ESTIMATORS = 2000

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

def genPlot(avgs, stds, title, fpad="../../results/plots/"):

    #st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M%S')
    fname = fpad + str(title) + '.png'

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
def genDuoPlot(avgs1, stds1, avgs2,stds2, title, fpad="../../results/plots/"):

    #st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M%S')
    fname = fpad + str(title) + '.png'

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


def step1(X,y, featureNames, threshold, runs=RUNS, criterion='gini'):
    print('step1')

    #get importances
    cart = DecisionTreeClassifier(criterion=criterion)
    importances = []
    for i in range(runs):
        cart.fit(X, y)
        importances.append(cart.feature_importances_)

    # get Average and std of the importances
    stds = np.std(importances, axis=0)
    importances = np.average(importances, axis=0)

    # genPlot(importances, stds, 'step1 importances'))

    #throw out everything <= threshold
    indices_to_keep = []
    for index, (imp, std) in enumerate(zip(importances, stds)):
        if imp > threshold: #worst case filtering
            indices_to_keep.append(index)
    indices_to_keep = np.array(indices_to_keep)

    #filter indices
    importances = importances[indices_to_keep]
    featureNames = featureNames[indices_to_keep]

    # sort features
    indices_to_keep = np.array(np.argsort(importances)[::-1])
    featureNames = featureNames[indices_to_keep]

    return np.array(indices_to_keep), featureNames
def step2_interpretation(X, y, featureNames, runs=RUNS, n_estimators=N_ESTIMATORS, criterion='gini'):
    print('step2_interpretation')
    featuresLeft = len(X[0])

    #for featCount = 1 ~> remaining indices
    oob_scores = []
    for featCount in range(1,featuresLeft + 1):
        print('inter - ' + str(featCount))
        run_errors = []
        forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features='auto',
            criterion=criterion,
            n_jobs=-1,
            oob_score=True,
            bootstrap=True
        )
        X_temp = X[:,:featCount]
        for i in range(runs):
            forest.fit(X_temp,y)
            run_errors.append(forest.oob_score_)

        oob_scores.append(run_errors)

    #std
    stds = np.std(oob_scores, axis=1)
    avgs = np.average(oob_scores, axis=1)

    #genPlot(avgs,stds,'oob_scores step2_interpretation')
    highest_avg_index = np.argmax(avgs)
    highest_avg       = avgs[highest_avg_index]
    highest_std       = stds[highest_avg_index]

    #look for smaller kandidates, keeping track of STD
    for i in range(1,highest_avg_index+1):
        if avgs[i] - stds[i] > highest_avg - highest_std:
            highest_avg_index = i
            highest_std = stds[i]
            highest_avg = avgs[i]

    return highest_avg_index +1, highest_avg, highest_std, avgs, stds
def step2_prediction(X, y, featureNames, runs=RUNS, n_estimators=N_ESTIMATORS, criterion='gini'):
    print('step2_prediction')
    featuresLeft = len(X[0])

    #for featCount = 1 ~> remaining indices
    best_features_to_keep = []
    best_score, best_std = 0, 0
    for feat in range(featuresLeft):
        print('pred - ' + str(feat))

        forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features='auto',
            criterion=criterion,
            n_jobs=-1,
            oob_score=True,
            bootstrap=True
        )

        # add new feature to the existing features
        features_to_keep = best_features_to_keep[:]
        features_to_keep.append(feat)

        #prepare the dataset
        X_temp = X[:,features_to_keep]

        #get scores
        run_scores = []
        for i in range(runs):
            forest.fit(X_temp,y)
            run_scores.append(forest.oob_score_)

        new_score = np.average(run_scores)
        new_std   = np.std(run_scores)

        #better?
        if new_score - new_std > best_score - best_std:
            best_score = new_score
            best_std   = new_std
            best_features_to_keep =features_to_keep

    return best_features_to_keep, best_score, best_std

def genReport(results):
    #results[person] = [
    #   [ featCount_inter  , score_inter, std_inter, featureNames_inter, avgs, stds ],
    #   [ len(indices_pred), score_pred , std_pred , featureNames_pred  ]
    #]

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
    f = open('../../results/CART_PERS_' + str(st) + ".csv", 'w')
    f.write("person;interScore;interStd;interCount;interFeat;\n")

    scores = []
    stds   = []
    for i in range(len(results[0])):
        methodScores = []
        methodSTDs   = []
        for person,result in enumerate(results):
            f.write(str(person) + ';')
            f.write(str(result[i][1]) + ';' + str(result[i][2]) + ';' + str(result[i][0]) + ';')
            for name in result[i][3]:
                f.write(str(name) + ';')

            f.write('\n')
            genPlot(result[0][4], result[0][5], 'method' + str(i) + ' oob score for person ' + str(person))

            methodScores.append(result[i][1])
            methodSTDs.append(result[i][2])
        genPlot(methodScores, methodSTDs, 'method' + str(i) + 'scores')
        scores.append(methodScores)
        stds.append(methodSTDs)

        f.write('\n')
        f.write('\n')

        f.write("person;predScore;predStd;predCount;predFeat;\n")
    f.close()

    genDuoPlot(scores[0], stds[0], scores[1], stds[1], 'interpretation vs perdiction scores')

def RFPerson(person):
    print('person: ' + str(person))

    to_ret = load('CART_P' + str(person))
    if to_ret == None:

        #load X , y
        # load all features & keep them in memory
        featExtr = getFeatures()
        featureNames = np.array(featExtr.getFeatureNames())

        y_cont = load('cont_y_p' + str(person))
        if y_cont == None:
            print('[Warn] Rebuilding cache -  person ' + str(person))
            X, y_cont = personLoader.NoTestsetLoader(
                classificator=Classificators.ContValenceClassificator(),
                featExtractor=featExtr,
            ).load(person)

            dump(X, 'X_p' + str(person))
            dump(y_cont, 'cont_y_p' + str(person))
        else:
            X = load('X_p' + str(person))

        y_disc = np.array(y_cont)
        y_disc[y_disc <= 5] = 0
        y_disc[y_disc >  5] = 1

        # manual Feature standardization
        X = X - np.average(X, axis=0)
        X = np.true_divide(X, np.std(X, axis=0))

        #step 1 determine importances using RF forest
        indices_step1, featureNames_step1 = step1(X,y_disc,featureNames,0.01)
        featureNames = np.array(featureNames_step1)
        indices = np.array(indices_step1)

        #filter features (X) based on the results from step 1
        X = X[:,indices]

        #step 2 - interpretation
        featCount_inter, score_inter, std_inter, avgs, stds = step2_interpretation(X, y_disc, featureNames)
        indices_inter = indices[:featCount_inter]
        featureNames_inter = featureNames[indices_inter]

        #step 2 - prediction
        indices_pred, score_pred, std_pred = step2_prediction(X, y_disc, featureNames)
        featureNames_pred = featureNames[indices_pred]

        to_ret = [
            [ featCount_inter  , score_inter, std_inter, featureNames_inter, avgs, stds ],
            [ len(indices_pred), score_pred , std_pred , featureNames_pred  ]
        ]

        dump(to_ret, 'CART_P' + str(person))

    print('[' + str(person) + 'interpretation - score: ' + str(to_ret[0][1]) + '(' + str(to_ret[0][2]) + ') with ' + str(to_ret[0][0]) +
          'prediction - score: ' + str(to_ret[1][1]) + ' (' + str(to_ret[1][2]) + ') with ' + str(to_ret[1][0])
          )


    return to_ret

if __name__ == '__main__':

    results = load('CART_pers_specific')
    if results == None:
        pool = Pool(processes=POOL_SIZE)
        results = pool.map(RFPerson, range(1, STOPPERSON+1))
        pool.close()
        pool.join()
        dump(results, 'CART_pers_specific')

    genReport(results)