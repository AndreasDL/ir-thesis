import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
matplotlib.rc('font', **font)


if __name__ == '__main__':
    pad = '../dataset'   
    names = ['valence', 'arousal', 'dominance', 'liking']

    v = [[],[],[],[]]
    
    bins = []
    bins.append([0] * 8)
    bins.append([0] * 8)
    bins.append([0] * 8)
    bins.append([0] * 8)

    for person in range(1,33):
        fname = str(pad) + '/s'
        if person < 10:
            fname += '0'
        fname += str(person) + '.dat'
        with open(fname,'rb') as f:
            p = pickle._Unpickler(f)
            p.encoding= ('latin1')
            data = p.load()
            #structure of data element:
            #data['labels'][video] = [valence, arousal, dominance, liking]
            #data['data'][video][channel] = [samples * 8064]
            for i in range(4):
                vals = np.array( data['labels'][:,i] ) #ATM only valence needed
                vals = (vals - 1) #1->9 to 0->8
                
                for val in vals:
                    bin = int(np.floor(val))
                    if bin == 8:
                        bin = 7
                    bins[i][bin] += 1

                v[i].extend(vals)
            
            matrix = []
            matrix.append([0] * 8)
            matrix.append([0] * 8)
            matrix.append([0] * 8)
            matrix.append([0] * 8)
            
            matrix.append([0] * 8)
            matrix.append([0] * 8)
            matrix.append([0] * 8)
            matrix.append([0] * 8)
            
            for i in range(len(data['labels'])):
                valence   = int(np.floor(data['labels'][i][0] - 1))
                dominance = int(np.floor(data['labels'][i][2] - 1))

                if valence == 8:
                    valence = 7

                if dominance == 8:
                    dominance = 7

                matrix[dominance][valence] += 1

        print('person ', person, ': sad->' , np.sum(bins[0][:3]), '\t\thappy->', np.sum(bins[0][4:]))
        for i in range(8):
            print('\t' , i , '->', bins[0][i], ' ', end ='')
        print()

        for i in range(8):
            print('\t' , i , '->', bins[2][i], ' ', end ='')
        print()

        bins = []
        bins.append([0] * 8)
        bins.append([0] * 8)
        bins.append([0] * 8)
        bins.append([0] * 8)

        if person == 14:
            pprint(matrix)
            exit()


    exit()
    print('global averages: ')
    for i in range(4):
        print('\t', names[i], np.mean(v[i]), '\t')
    
    
    for k in range(4):
        print('occurences ', names[k], ': ')
        for i in range(8):
            print('\t' , i , '->', bins[k][i])

    #create plots
    #2 x  2 plots
    fig, ax = plt.subplots(2,2)
    fig.subplots_adjust(hspace=.5)
    for j in range(2):
        for i in range(2):
            index = 2 * j + i
            ind = np.arange(8)
            ax[j,i].bar(ind,bins[index])
            ax[j,i].set_title(names[index])
    plt.show()
'''
    sad = np.sum(bins[0][:4])
    hap = np.sum(bins[0][4:])
    print(sad)
    print(hap)
'''