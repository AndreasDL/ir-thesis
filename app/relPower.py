import os
import pickle
import numpy as np
import models
from scipy.signal import butter, lfilter
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt


def getRelPower(samples):
    #foreach channel return alpha, beta, gamma, theta power (delta is neglected)
    startFreq = [8, 13, 30, 4, 4]
    stopFreq = [13, 30, 50, 8, 50] #last one = total
    retArr = []
    n = len(samples[0])
    Fs = 128
    nyq = 0.5 * Fs

    for j in range(32):
        powers = [None] * 5
        for band in range(len(startFreq)):

            #bandpass filter to get waveband
            low = startFreq[band] / nyq
            high = stopFreq[band] / nyq
            b, a = butter(6, [low, high], btype='band')
            sampl = lfilter(b, a, samples[j])

            #fft => get power components
            # fft computing and normalization
            Y = np.fft.fft(sampl)/n
            Y = Y[range(round(n/2))]

            #power squared within band
            power = 0
            for i in range(len(Y)):
                power += abs(Y[i]) **2
            power = np.sqrt(power)
            powers[band] = power

        power /= powers[-1] #last one is the total value
        
        retArr.extend(powers[:-1])

    #return
    return np.array(retArr)

if __name__ == "__main__":
    trainVideoCount = 40
    avg_test_err = 0
    avg_train_err = 0
    for person in range(1,32):

        fname = '../dataset/s'
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
            
            #init
            x_train = []
            y_train = np.zeros((trainVideoCount, 1)) # .. x 1 label
            x_test  = []
            y_test  = np.zeros((40-trainVideoCount, 1)) # .. x 1 label
        
            #load data
            y_train = data['labels'][:trainVideoCount,0] #only valence needed
            y_test  = data['labels'][trainVideoCount:,0]

            #split single person in test and train set
            for j in range(trainVideoCount): #for each video
                x_train.append( getRelPower(data['data'][j]) )
            
            for j in range( trainVideoCount, len(data['data']) ): #for each video
                x_test.append(  getRelPower(data['data'][j]) )

            #lin regression
            '''
            train_err, test_err, regr = models.linReg(np.array(x_train), y_train, None, None)
            print('\tP:', person, '\tmod: lin', 
                '\tTr err: ', train_err,
                '\tTe err: ' , test_err
            )'''

            #ridge regression
            train_err, test_err, regr = models.ridgeReg(np.array(x_train), y_train, None, None, cvSets=8)
            print('\tP:', person, '\tmod: ridge', 
                '\tTr err: ', train_err,
                '\tTe err: ' , test_err
            )

            train_sizes, train_scores, test_scores = learning_curve(regr, 
                x_train, y_train, 
                train_sizes=np.linspace(0.1, 1, 40),
                scoring="mean_squared_error", cv=8)

            #https://github.com/scikit-learn/scikit-learn/issues/2439
            plt.plot(train_sizes, -1 * train_scores.mean(1), 'o-', color="r")
            plt.plot(train_sizes, -1 * test_scores.mean(1) , 'o-', color="g")


            plt.xlabel("Train size")
            plt.ylabel("Mean Squared Error")
            plt.title('Learning curves - Red:train, green:test')
            plt.legend(loc="best")
            plt.show()

            exit()

            avg_test_err += test_err
            avg_train_err += train_err

    avg_test_err  /= 32
    avg_train_err /= 32
    print('avg Tr Err: ', avg_test_err, '\tavg Te err: ', avg_test_err)