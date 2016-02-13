import numpy as np
from sklearn.metrics import roc_auc_score
import datetime
import time

class AReporter:
    def __init__(self,featureExtractor):
        self.featExtr = featureExtractor #needed to get the feature names

    def optMetric(self,predictions,truths):
        return None

    def genReport(self,person,predictions,truths,fpad='../results/'):
        return None


class CSVReporter(AReporter):
    def __init__(self,featureExtractor):
        AReporter.__init__(self,featureExtractor)

    def optMetric(self,predictions,truths):
        return self.accuracy(predictions,truths)

    def genReport(self,person,predictions,truths,anova_out,fpad='../results/'):
        acc  = self.accuracy(predictions, truths)
        (tpr,tnr,fpr,fnr) = self.tprtnrfprfnr(predictions, truths)
        auc = self.auc(predictions, truths)

        used_features = anova_out.named_steps['anova'].get_support()

        best_k = 0
        for bool in used_features:
            if bool:
                best_k += 1

        #print to console
        print('person: ', person,
            ' - k: '  , str(best_k),
            ' - acc: ', str(acc),
            ' - tpr: ' , str(tpr),
            ' - tnr: ' , str(tnr),
            ' - auc: ', str(auc),
            'used features', anova_out.named_steps['anova'].get_support()
        )

        #output to file
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M%S')
        f = open(fpad + "output" + str(st) + ".txt", 'w')
        f.write('person;best_k;acc;tpr,tnr,fpr,fnr,auc\n')

        s = str(person+1) + ';' +\
            str(best_k) + ';' +\
            str(acc) + ';' +\
            str(tpr) + ';' + str(tnr) + ';' +\
            str(fpr) + ';' + str(fnr) + ';' +\
            str(auc) + ';'

        for bool in used_features:
            if bool:
                s += 'X'
            s+= ';'

        f.write(s)
        f.write('\n')
        f.close()

    def accuracy(self,predictions, truths):
        acc = 0
        for pred, truth in zip(predictions, truths):
            acc += (pred == truth)

        return acc / float(len(predictions))

    def tprtnrfprfnr(self,predictions, truths):
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

    def auc(self,predictions, truths):

        return roc_auc_score(truths, predictions)

    def classCorrect(self,predictions, truths):
        classCount = [0,0,0,0]
        for i in range(4):
            classCount[i] = np.sum(truths[truths == i])

        classPreds = [0,0,0,0]
        for pred, truth in zip(predictions, truths):
            if pred == truth:
                classPreds[int(truth)] += 1

        return np.array( np.array(classPreds) / np.array(classCount) )

    def dimCorrect(self,predictions, truths):
        #assign classes
        #              | low valence | high valence |
        #low  arrousal |      0      |       2      |
        #high arrousal |      1      |       3      |

        #dimCount = | low arr Count | high arr count | low valence count | high valence count |
        dimCount = [0,0,0,0]
        for i in range(2):
            #low (i==0) and high(i==1) arouousal
            dimCount[i]   = np.sum(truths[truths == i]) + np.sum(truths[truths == i+2])

            #low(i==0) and high (i==1) valence
            dimCount[i+2] = np.sum(truths[truths == i]) + np.sum(truths[truths == i+1])

        predCount = [0,0,0,0]
        for pred, truth in zip(predictions, truths):
            #arousal correct ?
            if pred == truth or pred == truth + 2 or pred == truth -2:
                #low (truth % 2 == 0) , high (truth % 2 == 1
                predCount[int(truth) % 2] += 1

            #valence
            if pred == truth or pred == truth + 1 or pred == truth -1:
                if truth == 0 or truth == 1:
                    predCount[2] += 1
                else:
                    predCount[3] += 1

        return np.array(predCount) / np.array(dimCount)
