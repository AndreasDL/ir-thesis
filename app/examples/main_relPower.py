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

def func(samples):
    features = []
    features.extend( FE.getRelPower(samples) )
    return features

def personWorker(person):
    #print('working on person ', person)
    #load dataset
    (X, y) = DL.loadSinglePersonData(
        classCount=classCount,
        borderDist=borderDist,
        featureFunc=func, #intervalFunc, #channelFunc
        person=person
    )

    #classify
    test_acc = models.linSVM(X, y)
    print('person: ', str(person), ' completed - test_acc: ', test_acc)
    return test_acc

def main_run():
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

    test_result = main_run()
    print(test_result)
    f.write("result: " + str(test_result) + "\n")

    f.close()