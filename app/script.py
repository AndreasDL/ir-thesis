import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from featureExtractor import calculateFeatures
from plotters import plot2D
import numpy as np
import pickle


with open('dataset/s01.dat','rb') as f:
    p = pickle._Unpickler(f)
    p.encoding= ('latin1')
    data = p.load()
    #structure of data element:
    #data['labels'][video , attribute]
    #data['data'][video, channel, value]
    for i in range(len(data['labels'])):
        y_train = data['labels'][i,1] #only valence needed

        #split single person in test and train set
        features = calculateFeatures(data['data'][i])
        print('valence: ', y_train, 
            "\n\tavg:", np.mean(features),
            "\n\tstd:", np.std(features)
        )
        
        plot2D(range(len(features)), features , 'interval', 'x/y', y_train)