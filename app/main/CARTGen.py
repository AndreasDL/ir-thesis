from custRF.randomforest.classification_forest import ClassificationForest
import numpy as np


def accuracy(predictions, truths):
    acc = 0
    for pred, truth in zip(predictions, truths):
        acc += (pred == truth)

    return acc / float(len(predictions))


if __name__ == '__main__':

    X, y = [], []
    for i in range(1000):
        X.append([i, 2*i, 3*i, 4*i])
        y.append(0)

    for i in range(1000,2000):
        X.append([i, 2 * i, 3 * i, 4 * i])
        y.append(1)

    X = np.array(X)
    y = np.array(y)


    rf = ClassificationForest(ntrees=20)
    rf.fit(X,y)
    pred = []
    for case in X:
        pred.append(rf.predict(case))

    print(accuracy(pred,y))