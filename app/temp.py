import numpy as np
import Classificators
from main.PERS.PersScript import PersScript
from main.PERS.RFPers import RFPers
from personLoader import load,dump


#appendix tables
def getAccs():
    temp = []
    for i in range(12):
        t = []
        for j in range(32):
            u = []
            for j in range(6):
                u.append([])
            t.append(u)
        temp.append(t)

    # get testaccs
    for featIndex, featSet in enumerate(['ALL','EEG','PHY']):
        for dimIndex, dim in enumerate([Classificators.ContValenceClassificator(), Classificators.ContArousalClassificator()]):
            model = PersScript(featSet, 32, 30, dim, 1, "D:/ir-thesis/dumpedData/persScript/")
            model.run()

            for person, data in enumerate(model.accs):
                for modelIndex, modelData in enumerate(data[0]):
                    temp[modelIndex][person][3*dimIndex+featIndex] = modelData[7]


    names = np.array(['Pearson R', 'Mutual Information', 'distance Correlation', 'Linear Regression', 'Lasso regression',
                      'Ridgde regression', 'SVM', 'Random Forest', 'STD',
                      'ANOVA', 'LDA', 'PCA'])

    f = open("temp.csv", 'w')
    for mindex, model in enumerate(temp):
        f.write('method;' + str(names[mindex]) + ';\n')
        f.write(';valence;;;arousal;;;\n')
        f.write('person;all;EEG;non-EEG;all;EEG;non-EEG;\n')

        for pindex, person in enumerate(model):
            f.write(str(pindex) + ';')
            for values in person:
                f.write(str(values) + ';')
            f.write('\n')
        f.write('\n\n\n')

    f.close()


if __name__ == '__main__':
    getAccs()
