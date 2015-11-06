import dataLoader as DL
import featureExtractor as FE
import models
import plotters
import util
import numpy as np
import pickle

if __name__ == "__main__":
    trainVideoCount = 36
    intervalLength = 2
    cvs = 2
    

    left_channels  = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'P3']
    right_channels = ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'P4']

    for i in range(len(left_channels)):
        left_channel  = FE.channelNames[left_channels[i]]
        right_channel = FE.channelNames[right_channels[i]]

        avg_test_err = 0
        avg_train_err = 0
        for person in range(1,32):
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
                    #log(left) - log(right)
                    left_values  = FE.getFrequencyPowerDensity('alpha', data['data'][j][left_channel],  intervalLength)
                    right_values = FE.getFrequencyPowerDensity('alpha', data['data'][j][right_channel], intervalLength)
                    x_train.append( np.log(right_values) - np.log(left_values) )

                for j in range( trainVideoCount, len(data['data']) ): #for each video
                    #log(left) - log(right)
                    left_values  = FE.getFrequencyPowerDensity('alpha', data['data'][j][left_channel],  intervalLength)
                    right_values = FE.getFrequencyPowerDensity('alpha', data['data'][j][right_channel], intervalLength)
                    x_test.append( np.log(right_values) - np.log(left_values))

                x_train = np.array(x_train)
                x_test  = np.array(x_test)
                #linear regression
                #train_err, test_err, regr = models.linReg(x_train,y_train,x_test,y_test)
                #print('\tmodel: linear',
                #    '\tTrain error: ', train_err,
                #    '\tTest error: ' , test_err
                #)

                #ridge regression
                train_err, test_err, regr = models.ridgeReg(x_train, y_train, x_test, y_test, cvSets=cvs)
                print('\tP:', person, '\tmod: ridge', 
                    '\tTr err: ', train_err,
                    '\tTe err: ' , test_err
                )

                avg_test_err += test_err
                avg_train_err += train_err

        avg_test_err  /= 32
        avg_train_err /= 32
        print('Channels: ', left_channels[i] , ' - ', right_channels[i], 
            '\tavg Tr Err: ', avg_test_err, 
            '\tavg Te err: ', avg_test_err
        )
