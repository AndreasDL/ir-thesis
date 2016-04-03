import personLoader
import Classificators
import featureExtractor as FE
import models
import reporters
import time
from multiprocessing import Pool
POOL_SIZE = 6

from personLoader import dump,load

def getFeatures():
    # create the features
    featExtr = FE.MultiFeatureExtractor()
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

    for front, post in zip(FE.all_frontal_channels, FE.all_frontal_channels):
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

    for channel in FE.all_phy_channels:
        featExtr.addFE(FE.AvgExtractor(channel, ''))
        featExtr.addFE(FE.STDExtractor(channel, ''))

    featExtr.addFE(FE.AVGHeartRateExtractor())
    featExtr.addFE(FE.STDInterBeatExtractor())

    return featExtr


def valenceWorker(person, criterion='entropy'):
    print('valence ' , person)
    #create the features
    featExtr = getFeatures()

    #create classificator
    classificator = Classificators.ValenceClassificator()

    #create personloader
    personLdr = personLoader.NoTestsetLoader(classificator,featExtr)

    #put in model
    model = models.RFClusterModel(personLdr)

    #run model
    results = model.run(person=person,criterion=criterion)

    return results
def arousalWorker(person, criterion='entropy'):
    print('arousal ' , person)
    #create the features
    featExtr = getFeatures()

    #create classificator
    classificator = Classificators.ArousalClassificator()

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
        results = pool.map(valenceWorker, range(1, 33))
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
        results = pool.map(arousalWorker, range(1, 33))
        pool.close()
        pool.join()
        dump(results,'arousal_importances_entropy_per_person')

    reporter.genReport(results)

    t2 = time.time()
    print("arousal complete, time spend: " + str(t2-t1))

    print("total time spend: " + str(t2-t0))