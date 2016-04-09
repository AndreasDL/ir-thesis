import personLoader
from personLoader import load, dump
import featureExtractor as FE
import Classificators

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

from multiprocessing import Pool
POOL_SIZE = 2

STOPPERSON = 32
RUNS = 20
N_ESTIMATORS = 1000
THRESHOLD = 0.05#featcount and samples
FOLDS = 5

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

def step1(X,y, featureNames, threshold, criterion='gini'):
    print('step1')

    #get importances
    forest = RandomForestClassifier(
        n_estimators=N_ESTIMATORS *3,
        max_features='auto',
        criterion=criterion,
        n_jobs=-1,
    )

    forest.fit(X,y)

    # get importances
    importances = forest.feature_importances_
    stds = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    importances -= stds

    # genPlot(importances, stds, 'step1 importances'))
    # sort features
    indices_to_keep = np.array(np.argsort(importances)[::-1])
    featureNames = featureNames[indices_to_keep]

    #keep upper %threshold
    indices_to_keep = indices_to_keep[:int(len(indices_to_keep)*threshold)]

    #filter indices
    importances = importances[indices_to_keep]
    featureNames = featureNames[indices_to_keep]

    # sort features
    indices_to_keep = np.array(np.argsort(importances)[::-1])
    featureNames = featureNames[indices_to_keep]

    return np.array(indices_to_keep), featureNames

def step2_interpretation(X, y, featureNames, runs=RUNS, n_estimators=N_ESTIMATORS, criterion='gini'):
    featuresLeft = len(X[0])
    print('step2_interpretation - featLeft: ' + str(featuresLeft))

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
    featuresLeft = len(X[0])
    print('step2_prediction - featLeft: ' + str(featuresLeft))

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
    f = open('../../results/RF_general' + str(st) + ".csv", 'w')
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

def fixStructure(all_X, all_y_disc):
    # structure of X
    X, y_disc = [], []
    for person_x, person_y in zip(all_X, all_y_disc):
        for video, label in zip(person_x, person_y):
            X.append(video)
            y_disc.append(label)

    X = np.array(X)
    y_disc = np.array(y_disc)

    return X, y_disc
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


    return persons_x, persons_y


if __name__ == '__main__':

    results = load('RF_general')
    if results == None:

        #load list[person][video][feature] = val
        featExtrs = getFeatures()
        featNames = np.array(featExtrs.getFeatureNames())

        all_y_disc = load('all_pers_y')
        if all_y_disc == None:
            print("[warn] rebuilding cache")
            personLoader = personLoader.PersonsLoader(
                classificator=Classificators.ContValenceClassificator, #will return disc values anyway!
                featExtractor=featExtrs,
                stopPerson=STOPPERSON
            )
            all_X, all_y_disc = personLoader.load()
            dump(all_y_disc, 'all_pers_y')
            dump(all_X, 'all_pers_X')
        else:
            all_X = load('all_pers_X')

        all_X = np.array(all_X)
        all_y_disc = np.array(all_y_disc)

        #train test split #TODO

        X, y_disc = fixStructure(all_X, all_y_disc)

        # manual Feature standardization
        X = X - np.average(X, axis=0)
        X = np.true_divide(X, np.std(X, axis=0))


        # step 1 determine importances using RF forest
        indices_step1, featureNames_step1 = step1(X, y_disc, featNames, THRESHOLD)
        featureNames = np.array(featureNames_step1)
        indices = np.array(indices_step1)

        # filter features (X) based on the results from step 1
        X = X[:, indices]

        #back to orig struct
        all_X, all_y_disc = reverseFixStructure(X,y_disc)

        # step 2 - interpretation
        featCount_inter, score_inter, std_inter, avgs, stds = step2_interpretation(all_X, all_y_disc, featureNames)
        indices_inter = indices[:featCount_inter]
        featureNames_inter = featureNames[indices_inter]

        # step 2 - prediction
        indices_pred, score_pred, std_pred = step2_prediction(all_X, all_y_disc, featureNames)
        featureNames_pred = featureNames[indices_pred]

        to_ret = [
            [featCount_inter, score_inter, std_inter, featureNames_inter, avgs, stds],
            [len(indices_pred), score_pred, std_pred, featureNames_pred]
        ]

        print('Interpretation - score: ' + str(to_ret[0][1]) + ' (' + str(to_ret[0][2]) + ') with ' + str(to_ret[0][0]) +
              '  prediction - score: ' + str(to_ret[1][1]) + ' (' + str(to_ret[1][2]) + ') with ' + str(to_ret[1][0])
              )

        dump(results, 'RF_general')

    genReport(results)