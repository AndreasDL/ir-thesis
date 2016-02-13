import personLoader
import classificators
import featureExtractor as FE
import models
import reporters

from multiprocessing import Pool
POOL_SIZE = 8

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



if __name__ == '__main__':

    #multithreaded
    pool = Pool(processes=POOL_SIZE)
    results = pool.map( valenceAnovaWorker, range(1,33) )
    pool.close()
    pool.join()


    reporter = reporters.CSVReporter()
    reporter.genReport(results)
