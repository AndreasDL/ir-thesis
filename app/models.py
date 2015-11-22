import numpy as np
from util import accuracy
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVC
from sklearn.cross_validation import KFold


def linSVM(X, y, CVSets=4):
    #use linear kernel
    clf = SVC(kernel='linear', probability=False) #true for later
    K_CV = KFold(len(X), n_folds=CVSets, random_state=17, shuffle=True)
    
    err = 0
    for train_index, CV_index in K_CV:
        clf.fit(X[train_index], y[train_index])

        #MSE train err
        err += accuracy(clf.predict(X[CV_index]), y[CV_index])

    test_acc = err / float(CVSets)
    return test_acc