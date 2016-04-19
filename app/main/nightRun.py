from Classificators import ContArousalClassificator, ContValenceClassificator
from main.PERS.RFPers import RFPers
from main.PERS.SVMPers import SVMPers
from main.PERS.PersScript import PersScript

if __name__ == '__main__':

    PersScript('PHY',2,30,ContArousalClassificator()).run()

    for model in [PersScript]:
        for feat in ['ALL','PHY','EEG']:
            for cls in [ContArousalClassificator(), ContValenceClassificator()]:
                try:
                    model(feat,32,30,cls).run()
                except Exception as e:
                    print(model, feat, cls, 'went wrong')
                    print(e)
