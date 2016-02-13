import datetime
import time
from multiprocessing import Pool

import models
import numpy as np

import personLoader as DL
from archive import featureExtractor as FE

all_left_channels  = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3']
all_right_channels = ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4']

testVideosRatio = 0.25
classCount = 2
borderDist = 0.25

used_interval=[1]
def intervalFunc(samples):
    features = []
    for interval in used_interval:
        features.extend( FE.LMinRFraction(samples,left_channel='F7', right_channel='F8', intervalLength=interval) )
    return features


def personWorker(person):
    #print('working on person ', person)
    #load dataset
    (X, y) = DL.loadSinglePersonData(
        classCount=classCount,
        borderDist=borderDist,
        featureFunc=intervalFunc,
        person=person
    )

    #classify
    test_acc = models.linSVM(X, y)
    return test_acc

   
def main_interval_search():
        pool = Pool(processes=8)
        results = pool.map( personWorker, range(1,33) )
        pool.close()
        pool.join()

        return np.mean(np.array(results))
     


if __name__ == "__main__":
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    f = open("output" + str(st) + ".txt", 'w')
    f.write(
        'testVideosRatio: ' + str(testVideosRatio) +
        ' classCount: ' + str(classCount) +
        ' borderDist: ' + str(borderDist) + "\n"
    )
    
    for interval in [1,2,5,10,15,20,25,30,35,40,45,50,55,60,63]:
        used_interval = [ interval ]

        print("interval - ", interval)

        test_result = main_interval_search()
        print(test_result)
        f.write('interval: ' + str(interval) + " result: " + str(test_result) + "\n")

    f.close()