import featureExtractor as FE
import personLoader
import models
import Classificators
import reporters

def getFeatures():
    # create the features
    featExtr = FE.MultiFeatureExtractor()

    #physiological signals
    '''
    for channel in FE.all_phy_channels:
        featExtr.addFE(FE.AvgExtractor(channel, ''))
        featExtr.addFE(FE.STDExtractor(channel, ''))
    '''
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

        featExtr.addFE(
            FE.FrontalMidlinePower(
                channels=[channel],
                featName="FM " + FE.all_channels[channel]
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
        featExtr.addFE(
            FE.LMinRLPlusRExtractor(
                left_channels=[left],
                right_channels=[right],
                featName='LR ' + FE.all_channels[left] + ',' + FE.all_channels[right]
            )
        )

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

def valenceWorker(criterion,treecount,threshold):

    featExtr = getFeatures()

    # create classificator
    classificator = Classificators.ValenceClassificator()

    # create personloader
    personLdr = personLoader.NoTestsetLoader(classificator, featExtr)

    # put in model
    model = models.RFModel(personLoader=personLdr,criterion=criterion,treeCount=treecount,threshold=threshold)

    # run model
    results = model.run()

    return results
def arousalWorker(criterion,treecount,threshold):
    featExtr = getFeatures()

    # create classificator
    classificator = Classificators.ArousalClassificator()

    # create personloader
    personLdr = personLoader.NoTestsetLoader(classificator, featExtr)

    # put in model
    model = models.RFModel(personLdr)

    # run model
    model = models.RFModel(personLoader=personLdr,criterion=criterion,treeCount=treecount,threshold=threshold)

    return results


if __name__ == '__main__':
    treeCount = 2000
    threshold = 0.0001

    reporter = reporters.HTMLRFModelReporter()

    results = valenceWorker('gini',treeCount,threshold)

    reporter.genReport(results)

    '''
    #reporter.genReport(results)
    results = valenceWorker('entropy',treeCount,threshold)
    #reporter.genReport(results)

    results = arousalWorker('gini',treeCount,threshold)
    #reporter.genReport(results)
    results = arousalWorker('entropy',treeCount,threshold)
    #reporter.genReport(results)
    '''