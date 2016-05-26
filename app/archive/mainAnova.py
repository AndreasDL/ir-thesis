import time
from multiprocessing import Pool

import classificators

import featureExtractor as FE
import personLoader
from archive import models, reporters

POOL_SIZE = 6

#use with CSV reporter
def getAnova():
    t0 = time.time()

    reporter = reporters.CSVReporter()

    #multithreaded
    pool = Pool(processes=POOL_SIZE)
    results = pool.map( valenceAnovaWorker, range(1,33) )
    pool.close()
    pool.join()
    reporter.genReport(results)

    t1 = time.time()
    print("valence complete, time spend: " + str(t1-t0))

    #multithreaded
    pool = Pool(processes=POOL_SIZE)
    results = pool.map( arousalAnovaWorker, range(1,33) )
    pool.close()
    pool.join()
    reporter.genReport(results)

    t2 = time.time()
    print("arousal complete, time spend: " + str(t2-t1))

    print("total time spend: " + str(t2-t0))
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
    model = models.StdModel(personLdr, 4)

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
    model = models.StdModel(personLdr, 4)

    #run model
    results = model.run(person)

    print('person: ' + str(person) + ' completed')
    return results

if __name__ == '__main__':
    getAnova()