import numpy as np
from sklearn.metrics import roc_auc_score

def accuracy(predictions, truths):
    acc = 0
    for pred, truth in zip(predictions, truths):
        acc += (pred == truth)

    return acc / float(len(predictions))

def tprtnrfprfnr(predictions, truths):
    tp, tn = 0, 0
    fp, fn = 0, 0
    
    pos_count = np.sum(truths)
    neg_count = len(truths) - pos_count
    
    for pred, truth in zip(predictions, truths):
        if pred == truth: #prediction is true
            if pred == 1:
                tp += 1
            else:
                tn += 1
        
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
        
    return tp / float(pos_count), tn / float(neg_count), fp / float(pos_count), fn / float(neg_count)

def auc(predictions, truths):

    return roc_auc_score(truths, predictions)