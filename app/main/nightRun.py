from .RFPers import RFPers
from .SVMPers import SVMPers
from Classificators import ContArousalClassificator, ContValenceClassificator

for model in [RFPers, SVMPers]:
    for feat in ['ALL','PHY','EEG']:
        for cls in [ContArousalClassificator(), ContValenceClassificator()]:
            model(feat,32,40,cls).run()