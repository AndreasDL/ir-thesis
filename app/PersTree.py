from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

from multiprocessing import Pool
POOL_SIZE = 10

class PersTree():
    def __init__(self,n_trees):
        self.n_trees = n_trees
        self.trees = []
        for n in range(n_trees):
            self.trees.append(DecisionTreeClassifier())

    def fit(self,X,y):
        self.oobErrs = []
        self.X = X
        self.y = y

        pool = Pool(processes=POOL_SIZE)
        self.oobErrs = pool.map( self.fitTree, range(self.n_trees))
        pool.close()
        pool.join()

    def fitTree(self,tree):
        persCount = len(self.X)
        for tree in self.trees:
            # create bootstrap & oob set
            indices_bootstrap = []
            for i in range(persCount):
                indices_bootstrap.append(random.randint(0, persCount - 1))

            indices_oob = [i for i in range(persCount)]
            for index in indices_bootstrap:
                if index in indices_oob:
                    indices_oob.remove(index)

            indices_bootstrap = np.array(indices_bootstrap)
            indices_oob = np.array(indices_oob)

            # create samples & fixStructure
            X_bootstrap, y_bootstrap = self.fixStructure(self.X[indices_bootstrap], self.y[indices_bootstrap])
            X_oob, y_oob = self.fixStructure(self.X[indices_oob], self.y[indices_oob])

            # train with bootstrap
            tree.fit(X_bootstrap, y_bootstrap)

            # get oob err
            return self.accuracy(tree.predict(X_oob), y_oob)

    def getOob(self):
        return np.average(self.oobErrs), np.std(self.oobErrs)

    def getImportance(self):
        importances = []
        for tree in self.trees:
            importances.append(tree.feature_importances_)

        return np.average(importances), np.std(importances)

    def fixStructure(self,all_X, all_y_disc):
        # structure of X
        X, y_disc = [], []
        for person_x, person_y in zip(all_X, all_y_disc):
            for video, label in zip(person_x, person_y):
                X.append(video)
                y_disc.append(label)

        X = np.array(X)
        y_disc = np.array(y_disc)

        return np.array(X), np.array(y_disc)

    def reverseFixStructure(self,X, y_disc):
        persons_x, persons_y = [], []
        person_x, person_y = [], []
        for index, (video, label) in enumerate(zip(X, y_disc)):

            if index % 40 == 0 and index != 0:  # start of a new person
                persons_x.append(person_x)
                persons_y.append(person_y)

                person_x = []
                person_y = []

            person_x.append(video)
            person_y.append(label)

        persons_x.append(person_x)
        persons_y.append(person_y)

        return np.array(persons_x), np.array(persons_y)

    def accuracy(self,predictions, truths):
        acc = 0
        for pred, truth in zip(predictions, truths):
            acc += (pred == truth)

        return acc / float(len(predictions))

