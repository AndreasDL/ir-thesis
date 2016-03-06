import personLoader
import Classificators
import featureExtractor as FE
import models
import reporters

def valenceCorrelationWorker(criterion):
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
    classificator = Classificators.ValenceClassificator()

    #create personloader
    personLdr = personLoader.PersonCombiner(classificator,featExtr)

    #put in model
    model = models.GlobalRFSelectionModel(personLdr)

    #run model
    results = model.run(criterion=criterion)

    return results
def arousalCorrelationWorker(criterion):
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
    classificator = Classificators.ArousalClassificator()

    #create personloader
    personLdr = personLoader.PersonCombiner(classificator,featExtr)

    #put in model
    model = models.GlobalRFSelectionModel(personLdr)

    #run model
    results = model.run(criterion=criterion)

    return results

if __name__ == '__main__':
    reporter = reporters.HTMLRFAnalyticsReporter()

    results = valenceCorrelationWorker('gini')
    reporter.genReport(results)
    results = valenceCorrelationWorker('entropy')
    reporter.genReport(results)

    results = arousalCorrelationWorker('gini')
    reporter.genReport(results)
    results = arousalCorrelationWorker('entropy')
    reporter.genReport(results)