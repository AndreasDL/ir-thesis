from Classificators import ContArousalClassificator, ContValenceClassificator
from main.PERS.RFPers import RFPers
from main.PERS.SVMPers import SVMPers

for model in [RFPers, SVMPers]:
    for feat in ['ALL','PHY','EEG']:
        for cls in [ContArousalClassificator(), ContValenceClassificator()]:
            try:
                model(feat,32,40,cls).run()
            except Exception as e:
                print(model, feat, cls, 'went wrong')
                print(e)
