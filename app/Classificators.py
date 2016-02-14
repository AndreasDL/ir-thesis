import numpy as np


class AClassificator:
    def __init__(self,name, classNames):
        self.name = name
        self.classNames = classNames

    def classify(self,labels):
        return []

    def getClassNames(self):
        return self.classNames

    def getName(self):
        return self.name

class FourClassificator(AClassificator):
    def __init__(self,classNames=['LVLA','LVHA','HVLA', 'HVHA'],name='fourClasses'):
        AClassificator.__init__(self,name,classNames)

    def classify(self,labels):
        #labels
        valences = np.array(labels[:,0]) #ATM only valence needed
        #valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1
        valences[ valences <= 5 ] = 0
        valences[ valences >  5 ] = 1

        arousals = np.array(labels[:,1])
        #arousals = (arousals - 1) / 8 #1->9 to 0->8 to 0->1
        arousals[ arousals <= 5 ] = 0
        arousals[ arousals >  5 ] = 1

        #assign classes
        #              | low valence | high valence |
        #low  arrousal |      0      |       2      |
        #high arrousal |      1      |       3      |

        y = np.zeros(len(valences))
        for i, (val, arr) in enumerate(zip(valences, arousals)):
            y[i] = (val * 2) + arr
        #TODO: probs with stratisfied shuffle split

        return y
class ValenceClassificator(AClassificator):
    def __init__(self,classNames=['LV','HV'],name='valenceClasses'):
        AClassificator.__init__(self,name,classNames)

    def classify(self,labels):
        #labels
        valences = np.array( labels[:,0] ) #ATM only valence needed
        #valences = (valences - 1) / 8 #1->9 to 0->8 to 0->1
        valences[ valences <= 5 ] = 0
        valences[ valences >  5 ] = 1

        #assign classes
        #low valence | high valence |
        #     0      |       1      |

        y = np.zeros(len(valences))
        for i, val in enumerate(valences):
            y[i] = val

        return y
class ArousalClassificator(AClassificator):
    def __init__(self,classNames=['LA','HA'],name='arousalClasses'):
        AClassificator.__init__(self,name,classNames)

    def classify(self,labels):
        #labels
        arousals = np.array( labels[:,1] )
        #arousals = (arousals - 1) / 8 #1->9 to 0->8 to 0->1
        arousals[ arousals <= 5 ] = 0
        arousals[ arousals >  5 ] = 1

        #assign classes
        #low  arrousal |      0      |
        #high arrousal |      1      |
        y = np.zeros(len(arousals))
        for i, arr in enumerate(arousals):
            y[i] = arr

        return y

class ContArousalClassificator(AClassificator):
    def __init__(self,name='ContArousalClasses'):
        AClassificator.__init__(self,name,[])

    def classify(self,labels):
        return np.array(labels[:,1])
class ContValenceClassificator(AClassificator):
    def __init__(self,name='ContValenceClasses'):
        AClassificator.__init__(self,name,[])

    def classify(self,labels):
        #labels
        return np.array(labels[:,0]) #valences