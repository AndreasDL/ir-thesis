import pickle

import models
import numpy as np

from archive import featureExtractor as FE

if __name__ == "__main__":
    trainVideoCount = 32
    person = 1
    intervalLength = 2
    cvs = 2
    pad = '../dataset/s'

    left_channels  = ['F3']#['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'P3']
    right_channels = ['F4']#['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'P4']


    #load data
    x_train = []
    y_train = np.zeros((trainVideoCount, 1)) # .. x 1 label
    x_test  = []
    y_test  = np.zeros((40-trainVideoCount, 1)) # .. x 1 label

    fname = str(pad)
    if person < 10:
        fname += '0'
    fname += str(person) + '.dat'
    with open(fname,'rb') as f:
        p = pickle._Unpickler(f)
        p.encoding= ('latin1')
        data = p.load()
        #structure of data element:
        #data['labels'][video , attribute]
        #data['data'][video, channel, value]

        y_train = data['labels'][:trainVideoCount,0] #only valence needed
        y_test  = data['labels'][trainVideoCount:,0]

        #split single person in test and train set
        for j in range(trainVideoCount): #for each video
            feat = []
            for i in range(len(left_channels)):

                left_channel  = FE.channelNames[left_channels[i]]
                right_channel = FE.channelNames[right_channels[i]]
                
                #reference all data to Cz
                left_samples  = data['data'][j][left_channel]  - data['data'][j][FE.channelNames['Cz']]
                right_samples = data['data'][j][right_channel] - data['data'][j][FE.channelNames['Cz']]

                #log(left) - log(right)
                left_values  = FE.getFrequencyPowerDensity('alpha', left_samples,  intervalLength)
                right_values = FE.getFrequencyPowerDensity('alpha', right_samples, intervalLength)
                feat.extend( np.log(left_values) - np.log(right_values) )
            x_train.append(feat)

        for j in range( trainVideoCount, len(data['data']) ): #for each video
            feat = []
            for i in range(len(left_channels)):
                left_channel  = FE.channelNames[left_channels[i]]
                right_channel = FE.channelNames[right_channels[i]]
                
                #reference all data to Cz
                left_samples  = data['data'][j][left_channel]  - data['data'][j][FE.channelNames['Cz']]
                right_samples = data['data'][j][right_channel] - data['data'][j][FE.channelNames['Cz']]

                #log(left) - log(right)
                left_values  = FE.getFrequencyPowerDensity('alpha', left_samples,  intervalLength)
                right_values = FE.getFrequencyPowerDensity('alpha', right_samples, intervalLength)
                feat.extend( np.log(left_values) - np.log(right_values) )
            x_test.append(feat)

        x_train = np.array(x_train)
        x_test  = np.array(x_test)

        #linear regression
        #train_err, test_err, regr = models.linReg(x_train,y_train,x_test,y_test)
        #print('\tmodel: linear',
        #    '\tTrain error: ', train_err,
        #    '\tTest error: ' , test_err
        #)

    #for cvs in range(2,8):
        #ridge regression
        train_err, test_err, regr = models.ridgeReg(x_train, y_train, x_test, y_test, cvSets=cvs)
        print('\tmodel: ridge', 'cv_Sets: ', cvs,
            '\tTrain error: ', train_err,
            '\tTest error: ' , test_err
        )