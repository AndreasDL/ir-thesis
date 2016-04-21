from Classificators import ContArousalClassificator, ContValenceClassificator
from main.PERS.RFPers import RFPers
from main.PERS.SVMPers import SVMPers
from main.PERS.PersScript import PersScript

if __name__ == '__main__':

    for model in [PersScript]:
        for cls in [ContValenceClassificator(), ContArousalClassificator()]:
            for feat in ['ALL','PHY','EEG']:
                try:
                    model(feat,32,30,cls).run()
                except Exception as e:
                    print(model, feat, cls, 'went wrong')
                    print(e)
