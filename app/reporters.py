import numpy as np
from sklearn.metrics import roc_auc_score
import datetime
import time
import operator

class AReporter:
    def genReport(self,results,fpad='../results/'):
        return None

    colorList = [
        '#ff0000', '#ff4000', '#ff8000', '#ffbf00', '#ffff00',
        '#bfff00', '#80ff00', '#40ff00', '#00ff00', '#00ff40',
    ]
    def getColor(self, value):
        #color list stolen with pride from http://www.w3schools.com/colors/colors_picker.asp

        index = int(value * len(self.colorList))
        return self.colorList[index]


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

class HTMLAnalyticsReporter(AReporter):

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
                <td>""" + str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')) + """</td>
            <tr>
            <tr>
                <td>Person Count</td>
                <td>""" + str(len(results)) + """</td>
            </tr>
            <tr></tr>
            <tr>
                <td>Abbr.</td>
                <td>Meaning</td>
            </tr>
            <tr>
                <td>AB</td>
                <td> Alpha /Beta ratio</td>
            </tr>
            <tr>
                <td>RL</td>
                <td> ( Right Alpha - left alpha ) / ( right alpha + left alpha )</td>
            </tr>
            <tr>
                <td>FM</td>
                <td> (frontal) theta power (frontal channels should perform better</td>
        </table>
        """)

        f.write('</br></br></br>')

        #color codes:
        f.write("Color codes and their corresponding correlation</br>")
        f.write("<table><tr>")
        for index, color in enumerate(self.colorList):
            startVal = round( index / len(self.colorList), 3)
            stopVal  = round( (index+1) /len(self.colorList) , 3)
            f.write("<td bgcolor=" + self.getColor(startVal) + ">" + str(startVal) + " - " + str(stopVal) + "</td>")
        f.write("</tr></table>")

        f.write('</br></br></br>')

        #correlation table:
        #|         | feat 1 | feeat 2 | ....
        #|person 1 | ....
        #|person 2 | ...
        corr_table = """
        <h1> Correlations </h1>
        <table>
            <tr>
                <td><b>Person</b></td>
        """

        overallFeatCount = dict()
        for featName in results[0]['feat_names']:
            fname=featName.replace(" ", "</br>")
            corr_table += '<td><b>' + str(fname) + '</b></td>'
            overallFeatCount[str(featName)] = 0

        corr_table += '</tr>'

        person_sections = "<h1>Person specific</h1>"

        for person, result in enumerate(results):
            #correlation table
            corr_table += "<tr>\n<td><b>" + str(person) + "</b></td>"

            #init arrays array = list<(corr, pval, name)> => | highest, lower, lower , lower, lowest |
            overallTop = [(0,0,0)] * 5
            eegTop = [(0,0,0)] * 5
            phyTop = [(0,0,0)] * 5

            #loop through results
            for r, featName in zip(result['feat_corr'], result['feat_names']):
                corr = abs(r[0]) #-1 ; +1 => we are only interested in the amount of correlation; whether it is positive or negative is something we don't care about
                pval =r[1]

                #fix correlation table
                corr_table += "<td bgcolor=" + str(self.getColor(corr)) + ">" + str(round(corr,3)) + ' (' + str(round(pval,3)) + ") </td>\n"

                #find
                #top 5 features
                i = 4
                while i > 0 and overallTop[i-1][0] < corr:
                    #move features back
                    overallTop[i] = overallTop[i-1]
                    #next iter
                    i -= 1
                overallTop[i] = (corr, pval, featName)

                if (featName[0:2] == 'A/' or featName[0:2] == 'LR' or featName[0:2] == 'FM'):
                    #top 5 EEG features
                    i = 4
                    while i > 0 and eegTop[i-1][0] < corr:
                        #move features back
                        eegTop[i] = eegTop[i-1]
                        #next iter
                        i -= 1
                    eegTop[i] = (corr, pval, featName)
                else:
                    #top 5 phy features
                    i = 4
                    while i > 0 and phyTop[i-1][0] < corr:
                        #move features back
                        phyTop[i] = phyTop[i-1]
                        #next iter
                        i -= 1
                    phyTop[i] = (corr, pval, featName)

            person_sections += "<h2>Person " + str(person) + "</h2>"
            person_sections += """
            <h3>overal top 5 </h3>
            <table>
                <tr>
                    <td><b>featname</b></td>
                    <td><b>correlation</b></td>
                    <td><b>pval</b<</td>
                </tr>
            """
            for winner in overallTop:
                person_sections += "<tr><td><b>" + str(winner[2]) + "</b></td><td bgcolor=" + self.getColor(winner[0]) + ">" + str(winner[0]) + "</td><td>" + str(winner[1]) + "</td></tr>"
                overallFeatCount[winner[2]] += 1
            person_sections += "</table></br></br>"

            person_sections += """
            <h3>eeg top 5 </h3>
            <table>
                <tr>
                    <td><b>featname</b></td>
                    <td><b>correlation</b></td>
                    <td><b>pval</b<</td>
                </tr>
            """
            for winner in eegTop:
                person_sections += "<tr><td><b>" + str(winner[2]) + "</b></td><td bgcolor=" + self.getColor(winner[0]) + ">" + str(winner[0]) + "</td><td>" + str(winner[1]) + "</td></tr>"
            person_sections += "</table></br></br>"

            person_sections += """
            <h3>phy top 5 </h3>
            <table>
                <tr>
                    <td><b>featname</b></td>
                    <td><b>correlation</b></td>
                    <td><b>pval</b<</td>
                </tr>
            """
            for winner in phyTop:
                person_sections += "<tr><td><b>" + str(winner[2]) + "</b></td><td bgcolor=" + self.getColor(winner[0]) + ">" + str(winner[0]) + "</td><td>" + str(winner[1]) + "</td></tr>"
            person_sections += "</table></br>/<br>"


            corr_table += "</tr>\n"
        corr_table += "</table>\n"

        #write to output
        f.write(corr_table)
        f.write('</br></br></br>')

        #featCount
        f.write("<h1>Feature Occurence in top 5</h1></br><table><tr><td>Feature</td><td>number of occurences in overall top 5</td></tr>")
        for key,value in sorted(overallFeatCount.items(), key=lambda x: x[1], reverse=True):
            if value == 0:
                break
            f.write("<tr><td>" + str(key) + "</td><td>" + str(value) + "</td></tr>")
        f.write("</table></br></br></br>")

        #person specific top 5 features
        f.write(person_sections)

        #close tags
        f.write("</body></html>")
        #close file
        f.close()

class CSVCorrReporter(CSVReporter):
    def genReport(self,results,fpad='../results/'):
        #input:
        '''{
            'feat_corr'         : featCorrelations,
            'feat_acc'          : featAccuracies,
            'test_acc'          : test_acc,
            'train_acc'         : best_acc,
            'best_k'            : best_k,
            'feat_names'        : featNames,
            'max_k'             : self.max_k
        }'''

        #output to file
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M%S')
        f = open(fpad + "output" + str(st) + ".txt", 'w')
        f.write('person;best_k;train_acc;test_acc;')

        for k in range(2,results[0]['max_k']):
            f.write('k=' + str(k) + ';')

        for i in range(results[0]['max_k']):
            f.write('#' + str(i) + ';')

        for name in results[0]['feat_names']:
            f.write(name + ';')
        f.write('\n')


        for person, result in enumerate(results):
            f.write(str(person) + ';' + str(result['best_k']) + ';' + str(result['train_acc']) + ';' + str(result['test_acc']) + ';')

            #results for different k values
            for acc in result['feat_acc']:
                f.write(str(round(acc,3)) + ';')

            #top max_k features
            for i in range(result['max_k']):
                map = result['feat_corr'][i]
                f.write(map['feat_name'] + '(' + str(map['feat_corr']) + ')' + ';')

            #sort feat_corr
            result['feat_corr'].sort(key=lambda tup: tup['feat_index'], reverse = True) #sort on index so featnames match

            #print all correlations
            for map in result['feat_corr']:
                f.write(str(map['feat_corr']) + ';')

            f.write('\n')

        f.close()

class HTMLCorrReporter(CSVReporter):
    colorList = [
    '#ff0000', '#ff4000', '#ff8000', '#ffbf00', '#ffff00',
    '#bfff00', '#80ff00', '#40ff00', '#00ff00', '#00ff40',
    ]
    def getColor(self, value):
        #color list stolen with pride from http://www.w3schools.com/colors/colors_picker.asp

        index = int(value * len(self.colorList))

        return self.colorList[index]

    def genReport(self,results,fpad='../results/'):
        #input:
        '''{
            'feat_corr'         : featCorrelations,
            'feat_acc'          : featAccuracies,
            'test_acc'          : test_acc,
            'train_acc'         : best_acc,
            'best_k'            : best_k,
            'feat_names'        : featNames,
            'max_k'             : self.max_k
        }'''

        #output to file
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M%S')
        f = open(fpad + "output" + str(st) + ".html", 'w')

                #write tags
        f.write("""<html>
            <head>
                <title>""" + str(results[0]['classificatorName']) + """ analytics</title>
            </head>
        <body>\n""")

        #overview table
        f.write("""<h1>overview</h1>
        <table>
            <tr>
                <td><b>Person</b></td>""")
        for k in range(2,results[0]['max_k'] + 1):
            f.write("<td><b>k=" + str(k)+"</b></td>")
        f.write("</td>")

        for person, result in enumerate(results):
            featAcc = result['feat_acc']
            max_k = result['max_k']
            #featCorr = result['feat_corr'][0:max_k]

            f.write('<tr><td><b>Person ' + str(person + 1) + '</b></td>')
            for k, acc in zip(range(2,max_k+1),featAcc):
                f.write('<td bgcolor=' + str(self.getColor(acc)) + ">" + str(acc) + "</td>")
            f.write("</tr>")

        f.write("</table></br></br></br>")

        f.write("<h1>Person Specific</h1>")
        for  person,result in enumerate(results):
            featAcc = result['feat_acc']
            max_k = result['max_k']
            featCorr = result['feat_corr'][0:max_k]

            f.write("<h2>Person" + str(person + 1) + '</h2><table><tr><td>')
            f.write("<img src=\"plots/person" + str(person+1) + ".png\" style=\"width:400px;height:300px;\" ></td><td>")

            f.write("""<table><tr>
                <td><b>what</b></td>""")
            for k in range(1,max_k+1):
                f.write('<td><b>k=' + str(k) + '</b></td>')

            f.write('</tr><tr><td><b>Accuracy</b></td><td></td>')

            for k, acc in zip(range(2,max_k+1),featAcc):
                f.write('<td bgcolor=' + str(self.getColor(acc)) + ">" + str(acc) + "</td>")

            f.write("</tr><tr><td><b>featNames</b></td>")

            for corr in featCorr: #featCorr sorted on correlation
                ''' corr = {
                    'feat_index' : index,
                    'feat_corr'  : corr[0],
                    'feat_name'  : featNames[index]
                    }'''

                f.write('<td>' + corr['feat_name'] + '</td>')

            f.write("</tr><tr><td><b>featCorr</b></td>")

            for corr in featCorr: #featCorr sorted on correlation
                ''' corr = {
                    'feat_index' : index,
                    'feat_corr'  : corr[0],
                    'feat_name'  : featNames[index]
                    }'''

                f.write('<td bgcolor=' + self.getColor(corr['feat_corr']) + '>' + str(corr['feat_corr']) + '</td>')
            f.write('</tr></table></td></table>')






        f.write("</html>")