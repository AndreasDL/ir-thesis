import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)


if __name__ == '__main__':
    pad = '../dataset'   
    names = ['valence', 'arousal', 'dominance', 'liking']
    
    print('person;0<->1;1<->2;2<->3;3<->4;4<->5;5<->6;6<->7;7<->8;avg;std;median;sad;happy')
    for person in range(1,33):
        
        v = [[],[],[],[]] #keep for calculating mean

        bins = []
        bins.append([0] * 8)
        bins.append([0] * 8)
        bins.append([0] * 8)
        bins.append([0] * 8)
        total_videos = float(32 * 40)

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
                vals = np.array( data['labels'][:,i] )
                vals = (vals - 1) #1->9 to 0->8
                
                for val in vals:
                    bin = int(np.floor(val))
                    if bin == 8:
                        bin = 7
                    bins[i][bin] += 1

                v[i].extend(vals)
        
        print(str(person), ';' , end='')
        for j in range(len(bins[0])):
            print( bins[0][j] , ';', end='' )

        print( np.mean(v[0]), ';', np.std(v[0]), ';', np.median(v[0]), ';', np.sum(bins[0][:4]), ';', np.sum(bins[0][4:]) )
    
    exit()
    print('global averages: ')
    for i in range(4):
        print('\t', names[i], np.mean(v[i]), '\t')
    
    bins = np.array(bins) / total_videos
    bins = bins * 100
    
    for k in range(4):
        print('occurences ', names[k], ': ')
        for i in range(8):
            print('\t' , i , '->', bins[k][i])

    #create plots
    #2 x  2 plots
    fig, ax = plt.subplots(2,2)
    fig.subplots_adjust(hspace=.5)

    ind = np.arange(8)
    
    for j in range(2):
        for i in range(2):
            index = 2 * j + i
            
            ax[j,i].bar(ind,bins[index])
            
            ax[j,i].set_title(names[index])
            ax[j,i].set_xlabel(names[index]  + ' value')
            ax[j,i].set_ylabel('percentage of videos')
            ax[j,i].set_xticklabels(('0', '0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875', '1'))

    plt.show()