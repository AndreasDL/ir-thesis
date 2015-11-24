from sklearn.metrics import roc_auc_score

def accuracy(predictions, truths):
    acc = 0
    for pred, truth in zip(predictions, truths):
        acc += (pred == truth)

    return acc / float(len(predictions))

def tptnfpfn(predictions, truths):
    tp, tn = 0, 0
    fp, fn = 0, 0

    for pred, truth in zip(predictions, truths):
        if pred == truth: #True
            if pred == 0: #negative
                tn += 1
            else: #positive
                tp += 1
        else: #false
            if pred == 0: #negative
                tn += 1
            else: #positive
                tp += 1

    return tp, tn, fp, fn

def auc(predictions, truths):
    return roc_auc_score(truths, predictions)