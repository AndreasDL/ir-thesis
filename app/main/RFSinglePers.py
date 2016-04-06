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


STOPPERSON = 3
RUNS = 5
N_ESTIMATORS = 100

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
    '''
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
    '''
    return featExtr
def genPlot(importances, std, fpad="../../results/plots/"):

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M%S')
    fname = fpad + 'plot_' + str(st) + '.png'

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances ")
    plt.bar(
        range(len(importances)),
        importances,
        color="r",
        yerr=std,
        align="center"
    )
    plt.xticks(range(0, len(importances), 50))
    plt.xlim([-1, len(importances)])
    plt.savefig(fname)
    plt.clf()

def step1(X,y, featureNames, runs=RUNS,n_estimators=N_ESTIMATORS, criterion='gini'):
    print('step1')

    # grow forest
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features='auto',
        criterion=criterion,
        n_jobs=-1,
    )

    importances = []
    for i in range(runs):
        # fit forest
        forest.fit(X, y)

        # get importances
        importances.append(forest.feature_importances_)
        # std within one run = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # get Average and std of the importances
    stds = np.std(importances, axis=0)
    importances = np.average(importances, axis=0)

    # genPlot(importances, stds)

    #get threshold from cart estimator
    cart = DecisionTreeClassifier(criterion=criterion)
    cart_importances = []
    for i in range(runs):
        cart.fit(X, y)
        cart_importances.append(cart.feature_importances_)
    thresh = float(np.amin(np.std(cart_importances, axis=0)))

    #throw out everything < threshold
    indices_to_keep = []
    for index, (imp, std) in enumerate(zip(importances, stds)):
        if imp - std > thresh: #worst case filtering
            indices_to_keep.append(index)
    indices_to_keep = np.array(indices_to_keep)

    #sort indices to keep
    importances = importances[indices_to_keep]
    featureNames = featureNames[indices_to_keep]

    # sort features
    indices_to_keep = np.array(np.argsort(importances)[::-1])
    featureNames = featureNames[indices_to_keep]

    return np.array(indices_to_keep), featureNames
def step2(X,y, featureNames, runs=RUNS,n_estimators=N_ESTIMATORS, criterion='gini'):
    print('step2')
    featuresLeft = len(X[0])

    #for featCount = 1 ~> remaining indices
    oob_scores = []
    for featCount in range(1,featuresLeft + 1):
        print(featCount)
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

    #genPlot(avgs,stds)

    highest_avg_index = np.argmax(avgs)
    highest_avg       = avgs[highest_avg_index]
    highest_std       = stds[highest_avg_index]

    #look for smaller kandidates, keeping track of STD
    for i in range(highest_avg_index):
        if avgs[i] - stds[i] > highest_avg - highest_std:
            highest_avg_index = i
            highest_std = stds[i]
            highest_avg = avgs[i]

    return highest_avg_index


def RFPerson(person):
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
    y_disc[y_disc > 5] = 1

    # manual Feature standardization
    X = X - np.average(X, axis=0)
    X = np.true_divide(X, np.std(X, axis=0))

    #step 1 determine importances using RF forest
    indices_step1 = load('indices')
    featureNames_step1  = load('featNames_step1')
    if indices_step1 == None or featureNames == None:
        indices_step1, featureNames_step1 = step1(X,y_disc,featureNames)
        dump(indices_step1, 'indices')
        dump(featureNames_step1, 'featNames_step1')

    featureNames = np.array(featureNames_step1)
    indices = np.array(indices_step1)

    #filter features (X) based on the results from step 1
    X = X[:,indices]

    #step 2
    featCount_step2 = step2(X,y_disc,featureNames_step1)
    indices = indices[:featCount_step2]
    X = X[:,indices]
    featureNames = featureNames[indices]





if __name__ == '__main__':
    #TODO for person in range(STOPPERSON):
    RFPerson(1)