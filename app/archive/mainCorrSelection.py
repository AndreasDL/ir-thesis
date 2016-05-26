import time
from multiprocessing import Pool

import Classificators
import featureExtractor as FE
import personLoader
from archive import models, reporters

POOL_SIZE = 6




#train CV add features, test acc
def getCorrelationAcc():
    t0 = time.time()

    reporter = reporters.HTMLCorrReporter()

    #multithreaded
    pool = Pool(processes=POOL_SIZE)
    results = pool.map( valenceCorrAccWorker, range(1,33) )
    pool.close()
    pool.join()
    reporter.genReport(results)

    t1 = time.time()
    print("valence complete, time spend: " + str(t1-t0))

    #multithreaded
    pool = Pool(processes=POOL_SIZE)
    results = pool.map( arousalCorrAccWorker, range(1,33) )
    pool.close()
    pool.join()
    reporter.genReport(results)

    t2 = time.time()
    print("arousal complete, time spend: " + str(t2-t1))

    print("total time spend: " + str(t2-t0))
def valenceCorrAccWorker(person):
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
    cont_classificator = Classificators.ContValenceClassificator()

    #create personloaders
    cont_personLdr = personLoader.NoTestsetLoader(cont_classificator,featExtr)

    #put in model
    model = models.CorrelationsSelectionModel(cont_personLdr)

    #run model
    results = model.run(person)

    return results
def arousalCorrAccWorker(person):
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
    cont_classificator = Classificators.ContArousalClassificator()

    #create personloaders
    cont_personLdr = personLoader.NoTestsetLoader(cont_classificator,featExtr)

    #put in model
    model = models.CorrelationsSelectionModel(cont_personLdr)

    #run model
    results = model.run(person)

    return results

if __name__ == '__main__':
    getCorrelationAcc()