import numpy as np
from sklearn.metrics import roc_auc_score
import datetime
import time

class AReporter:
    def genReport(self,results,fpad='../results/'):
        return None


class CSVReporter(AReporter):

    def genReport(self,results,fpad='../results/'):
        #output to file
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M%S')
        f = open(fpad + "output" + str(st) + ".txt", 'w')
        f.write('person;best_k;acc;tpr;tnr;fpr;fnr;auc;')

        feat_names  = results[0]['feat_names']
        for feat_name in feat_names:
            f.write(feat_name)
            f.write(';')
        f.write('\n')


        for person, result in enumerate(results):
            predictions = result['predictions']
            truths      = result['truths']
            feat_list   = result['feat_list']

            acc  = self.accuracy(predictions, truths)
            (tpr,tnr,fpr,fnr) = self.tprtnrfprfnr(predictions, truths)
            auc = self.auc(predictions, truths)

            best_k = 0
            for bool in feat_list:
                if bool:
                    best_k += 1

            #print to console
            print('person: ', person,
                ' - k: '  , str(best_k),
                ' - acc: ', str(acc),
                ' - tpr: ' , str(tpr),
                ' - tnr: ' , str(tnr),
                ' - auc: ', str(auc),
                #'used features', feat_list()
            )




            s = str(person+1) + ';' +\
                str(best_k) + ';' +\
                str(acc) + ';' +\
                str(tpr) + ';' + str(tnr) + ';' +\
                str(fpr) + ';' + str(fnr) + ';' +\
                str(auc) + ';'

            for bool in feat_list:
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

class AnalyticsReporter(AReporter):
    def getColor(self, value):
        #color list stolen with pride from http://www.w3schools.com/colors/colors_picker.asp
        colorList = [
            '#ff0000', '#ff4000', '#ff8000', '#ffbf00', '#ffff00',
            '#bfff00', '#80ff00', '#40ff00', '#00ff00', '#00ff40',
        ]
        index = int(value * len(colorList))
        return colorList[index]


    def genReport(self,results,fpad='../results/'):

        #gen file
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M%S')
        f = open(fpad + "Analytics" + str(results[0]['classificatorName']) + str(st) + ".html", 'w')

        #write tags
        f.write("""<html>
            <head>
                <title>""" + str(results[0]['classificatorName']) + """ analytics</title>
            </head>
        <body>\n""")

        #meta table
        f.write("""<h1>Meta Data</h1>
        <table>
            <tr>
                <td>Classificator Used:</td>
                <td>""" + str(results[0]['classificatorName']) + """</td>
            </tr>
            <tr>
                <td>Created on</td>
                <td>""" + str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:M:S')) + """</td>
            <tr>
            <tr>
                <td>Person Count</td>
                <td>""" + str(len(results)) + """</td>
            </tr>
        </table>
        """)

        f.write('</br></br></br>')


        #correlation table:
        #|         | feat 1 | feeat 2 | ....
        #|person 1 | ....
        #|person 2 | ...
        corr_table = """
        <h1> Correlations </h1>
        <table>
            <tr>
                <td>Person</td>
        """
        for featName in results[0]['feat_names']:
            corr_table += '<td>' + str(featName) + '</td>'
        corr_table += '</tr>'

        #person specific table
        '''
        pers_tables = """
        <h1> Person Specific </h1>
        """
        '''

        for person, result in enumerate(results):
            #correlation table
            corr_table += "<tr>\n<td>" + str(person) + "</td>"
            for r in result['feat_corr']:
                corr = abs(r[0]) #-1 ; +1 => we are only interested in the amount of correlation; whether it is positive or negative is something we don't care about
                pval =r[1]
                corr_table += "<td bgcolor=" + str(self.getColor(corr)) + ">" + str(round(corr,3)) + ' (' + str(round(pval,3)) + ") </td>\n"
            corr_table += "</tr>\n"

            #person specific table
            '''
            pers_tables += "<h2> Person " + str(person) + " </h2>\n<table><tr><td>Feat</td><td>Accuracy</td></tr>\n"
            for acc, featName in zip(results[person]['feat_acc'], results[person]['feat_names']):
                pers_tables += "<tr><td>featName</td><td bgcolor=" + str(self.getColor(acc)) + ">" + str(round(acc,3)) + "</td></tr>"
            pers_tables += "</table></br></br>"
            '''

        corr_table += "</table>\n"

        #write to output
        f.write(corr_table)
        f.write('</br></br></br>')
        #f.write(pers_tables)

        #close tags
        f.write("</body></html>")
        #close file
        f.close()