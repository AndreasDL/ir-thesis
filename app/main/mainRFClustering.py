import personLoader
import classificators
import featureExtractor as FE
import models
import reporters
import time
from multiprocessing import Pool
POOL_SIZE = 6

from personLoader import dump,load



def valenceCorrelationWorker(person,criterion='entropy'):
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
    classificator = classificators.ValenceClassificator()

    #create personloader
    personLdr = personLoader.NoTestsetLoader(classificator,featExtr)

    #put in model
    model = models.RFClusterModel(personLdr)

    #run model
    results = model.run(person=person,criterion=criterion)

    return results
def arousalCorrelationWorker(person,criterion='entropy'):
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
    classificator = classificators.ArousalClassificator()

    #create personloader
    personLdr = personLoader.NoTestsetLoader(classificator,featExtr)

    #put in model
    model = models.RFClusterModel(personLdr)

    #run model
    results = model.run(person=person,criterion=criterion)

    return results

if __name__ == '__main__':

    reporter = reporters.HTMLRFClusteringReporter()

    t0 = time.time()

    #multithreaded
    results = load('valence_importances_entropy_per_person')
    if results == None:
        print("[warn] rebuilding cache")
        pool = Pool(processes=POOL_SIZE)
        results = pool.map( valenceCorrelationWorker, range(1,33) )
        pool.close()
        pool.join()
        dump(results,'valence_importances_entropy_per_person')

    reporter.genReport(results)

    t1 = time.time()
    print("valence complete, time spend: " + str(t1-t0))

    #multithreaded
    results = load('arousal_importances_entropy_per_person')
    if results == None:
        print("[warn] rebuilding cache")
        pool = Pool(processes=POOL_SIZE)
        results = pool.map( arousalCorrelationWorker, range(1,33) )
        pool.close()
        pool.join()
        dump(results,'arousal_importances_entropy_per_person')

    reporter.genReport(results)

    t2 = time.time()
    print("arousal complete, time spend: " + str(t2-t1))

    print("total time spend: " + str(t2-t0))