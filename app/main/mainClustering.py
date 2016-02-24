import personLoader
import classificators
import featureExtractor as FE
import models

import time

from personLoader import dump, load
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path


#use with analytics reporter
def getAnalytics():
    t0 = time.time()

    feat_corr = load('valence_feat_corr')
    if feat_corr == None:
        feat_corr = valenceCorrelationWorker()
        dump(feat_corr,'valence_feat_corr')

    #here feat_corr has a list of correlations of each feature for each person => dim reduc & 3D plot
    X_3D = PCA(n_components=3).fit_transform(feat_corr)
    X_2D = PCA(n_components=2).fit_transform(feat_corr)

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2])
    plt.show()

    plt.clf()
    plt.scatter(X_2D[:, 0], X_2D[:, 1])
    plt.show()

    t1 = time.time()
    print("valence complete, time spend: " + str(t1-t0))

    feat_corr = load('arousal_feat_corr')
    if feat_corr == None:
        feat_corr = arousalCorrelationWorker()
        dump(feat_corr,'arousal_feat_corr')

    #here feat_corr has a list of correlations of each feature for each person => dim reduc & 3D plot
    X_3D = PCA(n_components=3).fit_transform(feat_corr)
    X_2D = PCA(n_components=2).fit_transform(feat_corr)

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2])
    plt.show()

    plt.clf()
    plt.scatter(X_2D[:, 0], X_2D[:, 1])
    plt.show()

    t2 = time.time()
    print("arousal complete, time spend: " + str(t2-t1))

    print("total time spend: " + str(t2-t0))
def valenceCorrelationWorker():
    #create the features
    featExtr = FE.MultiFeatureExtractor()
    for channel in FE.all_EEG_channels:
        featExtr.addFE(
            FE.AlphaBetaExtractor(
                channels=[channel],
                featName='A/B ' + FE.all_channels[channel]
            )
        )
    for left, right in zip(FE.all_left_channels, FE.all_right_channels):
        featExtr.addFE(
            FE.LMinRLPlusRExtractor(
                left_channels=[left],
                right_channels=[right],
                featName='LR ' + FE.all_channels[left] + ',' + FE.all_channels[right]
            )
        )

    for channel in FE.all_EEG_channels:
        featExtr.addFE(
            FE.FrontalMidlinePower(
                channels=[channel],
                featName="FM " + FE.all_channels[channel]
            )
        )

    for channel in FE.all_phy_channels:
        featExtr.addFE(FE.AvgExtractor(channel,''))
        featExtr.addFE(FE.STDExtractor(channel,''))

    featExtr.addFE(FE.AVGHeartRateExtractor())
    featExtr.addFE(FE.STDInterBeatExtractor())

    #create classificator
    classificator = classificators.ContValenceClassificator()

    #create personloader
    personLdr = personLoader.NoTestsetLoader(classificator,featExtr)

    #put in model
    model = models.CorrelationsClusteringsModel(personLdr)

    #run model
    results = model.run()

    return results
def arousalCorrelationWorker():
    #create the features
    featExtr = FE.MultiFeatureExtractor()
    for channel in FE.all_EEG_channels:
        featExtr.addFE(
            FE.AlphaBetaExtractor(
                channels=[channel],
                featName='A/B ' + FE.all_channels[channel]
            )
        )
    for left, right in zip(FE.all_left_channels, FE.all_right_channels):
        featExtr.addFE(
            FE.LMinRLPlusRExtractor(
                left_channels=[left],
                right_channels=[right],
                featName='LR ' + FE.all_channels[left] + ',' + FE.all_channels[right]
            )
        )

    for channel in FE.all_EEG_channels:
        featExtr.addFE(
            FE.FrontalMidlinePower(
                channels=[channel],
                featName="FM " + FE.all_channels[channel]
            )
        )

    for channel in FE.all_phy_channels:
        featExtr.addFE(FE.AvgExtractor(channel,''))
        featExtr.addFE(FE.STDExtractor(channel,''))

    featExtr.addFE(FE.AVGHeartRateExtractor())
    featExtr.addFE(FE.STDInterBeatExtractor())

    #create classificator
    classificator = classificators.ContArousalClassificator()

    #create personloader
    personLdr = personLoader.NoTestsetLoader(classificator,featExtr)

    #put in model
    model = models.CorrelationsClusteringsModel(personLdr)

    #run model
    results = model.run()

    return results

if __name__ == '__main__':
    getAnalytics()