import pickle
import numpy as np
import os.path
from sklearn.cross_validation import StratifiedShuffleSplit

DATASET_LOCATION = "C:/dataset"
from multiprocessing import Pool
POOL_SIZE = 3


class APersonLoader:
    def __init__(self, classificator, featExtractor, name, path=DATASET_LOCATION):
        self.name = name
        self.path = path


        self.classificator = classificator
        self.featureExtractor  = featExtractor

    def load(self,person):
        #return X_train, y_train, x_test, y_test
        return [], [], [], []
class PersonLoader(APersonLoader):
    def __init__(self, classificator, featExtractor, name='normal', path=DATASET_LOCATION ):
        APersonLoader.__init__(self, classificator, featExtractor, name, path=DATASET_LOCATION)

    def load(self,person):
        fname = str(self.path) + '/s'
        if person < 10:
            fname += '0'
        fname += str(person) + '.dat'
        with open(fname,'rb') as f:
            p = pickle._Unpickler(f)
            p.encoding= ('latin1')
            data = p.load()

            #structure of data element:
            #data['labels'][video] = [valence, arousal, dominance, liking]
            #data['data'][video][channel] = [samples * 8064]


            X = self.featureExtractor.extract(data['data'])
            y = self.classificator.classify(data['labels'])

            #split train / test
            #n_iter = 1 => abuse the shuffle split, to obtain a static break, instead of crossvalidation
            sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=19)
            for train_set_index, test_set_index in sss:
                X_train, y_train = X[train_set_index], y[train_set_index]
                X_test , y_test  = X[test_set_index] , y[test_set_index]

            #fit normalizer to train set & normalize both train and testset
            #normer = Normalizer(copy=False)
            #normer.fit(X_train, y_train)
            #X_train = normer.transform(X_train, y_train, copy=False)
            #X_test  = normer.transform(X_test, copy=False)

            return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
class NoTestsetLoader(APersonLoader):
    def __init__(self, classificator, featExtractor, name='noTestset', path=DATASET_LOCATION ):
        APersonLoader.__init__(self, classificator, featExtractor, name, path=path)

    def load(self,person):
        fname = str(self.path) + '/s'
        if person < 10:
            fname += '0'
        fname += str(person) + '.dat'
        with open(fname,'rb') as f:
            p = pickle._Unpickler(f)
            p.encoding= ('latin1')
            data = p.load()

            #structure of data element:
            #data['labels'][video] = [valence, arousal, dominance, liking]
            #data['data'][video][channel] = [samples * 8064]


            X = self.featureExtractor.extract(data['data'])
            y = self.classificator.classify(data['labels'])

            return np.array(X), np.array(y)
class PersonCombiner(APersonLoader):
    def __init__(self, classificator, featExtractor, name='normal', path=DATASET_LOCATION ):
        APersonLoader.__init__(self, classificator, featExtractor, name, path=DATASET_LOCATION)

    def load(self, personList=range(1,33)):
        X, y = [], []

        for person in personList:
            print('loading person ' + str(person))
            fname = str(self.path) + '/s'
            if person < 10:
                fname += '0'
            fname += str(person) + '.dat'
            with open(fname,'rb') as f:
                p = pickle._Unpickler(f)
                p.encoding= ('latin1')
                data = p.load()

                #structure of data element:
                #data['labels'][video] = [valence, arousal, dominance, liking]
                #data['data'][video][channel] = [samples * 8064]

                X.extend(self.featureExtractor.extract(data['data']))
                y.extend(self.classificator.classify(data['labels']))

        return np.array(X), np.array(y)

class PersonsLoader(APersonLoader):
    def __init__(self, classificator, featExtractor, stopPerson, name='persTestset', path=DATASET_LOCATION):
        APersonLoader.__init__(self, classificator, featExtractor, name, path=path)

        self.stopPerson = stopPerson + 1 #+1 since the first person is person 1 and not person 0
        self.ldr = NoTestsetLoader(classificator,featExtractor)

    def load(self):
        pool = Pool(processes=POOL_SIZE)
        temp_data = pool.map(self.loadPerson, range(1, self.stopPerson)) #+1 done at init
        pool.close()
        pool.join()

        #omzetten naar arrays
        X, y = [], []
        for dct in temp_data:
            X.append(dct['X'])
            y.append(dct['y'])

        return np.array(X), np.array(y)

    def loadPerson(self,person):
        print('loading person ' + str(person))

        y = load('cont_y_p' + str(person))
        y_disc = None
        if y == None:
            print('[warn] rebuilding cache for person ' + str(person))
            X, y = self.ldr.load(person)
            dump(X, 'X_p' + str(person))
            dump(y, 'cont_y_p' + str(person))
        else:
            X = load('X_p' + str(person))

        # to disc
        y_disc = np.array(y)
        y_disc[y_disc <= 5] = 0
        y_disc[y_disc > 5] = 1

        return {'X':X, 'y': y_disc} #not logic since you give cont class, but hey it works



def dump(X, name, path='../../dumpedData'):
    fname = path + '/' + name
    with open(fname, 'wb') as f:
        pickle.dump( X, f )

def load(name, path='../../dumpedData'):
    fname = path + '/' + name
    #print('loading from')
    #print(os.path.abspath(fname))

    data = None
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            p = pickle._Unpickler(f)
            p.encoding= ('latin1')
            data = p.load()

    return data