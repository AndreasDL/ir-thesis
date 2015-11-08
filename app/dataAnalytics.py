import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 15}

    matplotlib.rc('font', **font)

    for j in range(2):
        for i in range(2):
            index = 2 * j + i
            ind = np.arange(8)
            ax[j,i].bar(ind,bins[index])
            ax[j,i].set_title(names[index])
    plt.show()