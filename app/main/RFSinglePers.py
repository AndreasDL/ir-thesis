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

def step1(X,y, features, runs=RUNS,n_estimators=N_ESTIMATORS, criterion='gini'):
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
    importances = []
    for i in range(runs):
        cart.fit(X, y)
        importances.append(cart.feature_importances_)
    std = np.std(importances, axis=0)
    # importances = np.average(importances, axis=0)
    thresh = float(np.amin(std))

    #throw out everything < threshold
    indices_to_keep = []
    for index, (imp, std) in enumerate(zip(importances, stds)):
        if float(imp) - float(std) > thresh: #worst case filtering
            indices_to_keep.append(index)



def RFPerson(person):
    #load X , y
    # load all features & keep them in memory
    y_cont = load('cont_y_p' + str(person))
    featExtr = getFeatures() #needed later on
    if y_cont == None:
        print('[Warn] Rebuilding cache -  person ' + str(person))
        X, y_cont = personLoader.PersonsLoader(
            classificator=Classificators.ContValenceClassificator(),
            featExtractor=featExtr,
            stopPerson=STOPPERSON,
        ).load()

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
    step1(X,y_disc,featExtr.getFeatureNames())


    print('hoi')



if __name__ == '__main__':
    RFPerson(1)