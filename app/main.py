import personLoader
import classificators
import featureExtractor as FE
import models
import reporters

from multiprocessing import Pool
POOL_SIZE = 8

#use with CSV reporter
def valenceAnovaWorker(person):
    #create the features
    featExtr = FE.MultiFeatureExtractor()
    featExtr.addFE(FE.AlphaBetaExtractor(FE.all_EEG_channels))
    featExtr.addFE(FE.LMinRLPlusRExtractor(FE.all_left_channels,FE.all_right_channels))
    featExtr.addFE(FE.FrontalMidlinePower(FE.all_FM_channels))

    for channel in FE.all_phy_channels:
        featExtr.addFE(FE.AvgExtractor(channel,''))
        featExtr.addFE(FE.STDExtractor(channel,''))

    featExtr.addFE(FE.AVGHeartRateExtractor())
    featExtr.addFE(FE.STDInterBeatExtractor())

    #create classificator
    classificator = classificators.ValenceClassificator()

    #create personloader
    personLdr = personLoader.PersonLoader(classificator,featExtr)

    #put in model
    model = models.StdModel(personLdr,4)

    #run model
    results = model.run(person)

    print('person: ' + str(person) + ' completed')
    return results
def arousalAnovaWorker(person):
    #create the features
    featExtr = FE.MultiFeatureExtractor()
    featExtr.addFE(FE.AlphaBetaExtractor(FE.all_EEG_channels))
    featExtr.addFE(FE.LMinRLPlusRExtractor(FE.all_left_channels,FE.all_right_channels))
    featExtr.addFE(FE.FrontalMidlinePower(FE.all_FM_channels))

    for channel in FE.all_phy_channels:
        featExtr.addFE(FE.AvgExtractor(channel,''))
        featExtr.addFE(FE.STDExtractor(channel,''))

    featExtr.addFE(FE.AVGHeartRateExtractor())
    featExtr.addFE(FE.STDInterBeatExtractor())

    #create classificator
    classificator = classificators.ArousalClassificator()

    #create personloader
    personLdr = personLoader.PersonLoader(classificator,featExtr)

    #put in model
    model = models.StdModel(personLdr,4)

    #run model
    results = model.run(person)

    print('person: ' + str(person) + ' completed')
    return results

#use with analytics reporter
def valenceCorrelationWorker(person):
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
                featName='L-R/L+R ' + FE.all_channels[left] + ',' + FE.all_channels[right]
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
    model = models.CorrelationsSelectionModel(personLdr)

    #run model
    results = model.run(person)

    return results
def arousalCorrelationWorker(person):
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
                featName='L-R/L+R ' + FE.all_channels[left] + ',' + FE.all_channels[right]
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
    model = models.CorrelationsSelectionModel(personLdr)

    #run model
    results = model.run(person)

    return results


if __name__ == '__main__':
    reporter = reporters.AnalyticsReporter()

    #multithreaded
    pool = Pool(processes=POOL_SIZE)
    results = pool.map( valenceCorrelationWorker, range(1,33) )
    pool.close()
    pool.join()

    reporter.genReport(results)

    #multithreaded
    pool = Pool(processes=POOL_SIZE)
    results = pool.map( arousalCorrelationWorker, range(1,33) )
    pool.close()
    pool.join()
    reporter.genReport(results)