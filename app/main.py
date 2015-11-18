import dataLoader as DL
import featureExtractor as FE
import models
from multiprocessing import Pool

left_channels_to_try  = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3']
right_channels_to_try = ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4']
testVideos = 8
classCount = 2

def channelPairWorker(index):
    left  = left_channels_to_try[index]
    right = right_channels_to_try[index]
    print("running channel pair: ", left, " - ", right)

    avg_test, avg_train = 0, 0

    #generate the extraction function
    def func(samples):
        return FE.LMinRFraction(samples, left_channel=left, right_channel=right)

    for person in range(1,33):
        #load dataset
        (X_train, y_train, X_test, y_test) = DL.loadSinglePersonData(classCount=classCount, testVideos=testVideos, featureFunc=func, person=person)

        #classify
        train_acc, test_acc, clf = models.linSVM(X_train,y_train, X_test,y_test)
        avg_test  += test_acc
        avg_train += train_acc

        #scores for one person
        #print('Person: ', person,
        #    '\tmodel: lin SVM',
        #    '\tTrain accuracy: ', train_acc,
        #    '\tTest accuracy: ' , test_acc
        #)

    avg_test  /= 32
    avg_train /= 32

    return test_acc


def main_all_one_by_one():
    f = open('output-noborder.txt', 'w')
    f.write('testVideos: ' + str(testVideos) + ' classCount: ' + str(classCount) + "\n")

    pool = Pool(processes=5)
    results = pool.map( channelPairWorker, range(len(left_channels_to_try)) )
    
    for i, result in enumerate(results):
    	print(left_channels_to_try[i], ' - ', right_channels_to_try[i], ' -> ', result)

if __name__ == "__main__":
    main_all_one_by_one()