import random

import matplotlib
import matplotlib.pylab as plt
import numpy as np
from sklearn import linear_model

import Classificators
from PersScript import PersScript
from personLoader import load

font = {'family': 'normal',
#        'weight': 'bold',
        'size': 25}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.titlesize'] = 25
matplotlib.rcParams['axes.labelsize'] = 25

def genPlot(avgs,stds,lbls,title,xLbl= '', yLbl='',bar_colors=None,fpad="../results/plots/", type='normal'):

    fname = fpad + 'accComp_' + str(title) + '.png'

    # Plot the feature importances of the forest
    fig, ax = plt.subplots()
    fig.set_size_inches(15,9)

    plt.title(title)
    for i, (avg, std) in enumerate(zip(avgs,stds)):

        color = "b"
        if bar_colors != None:
            color = bar_colors[i]

        ax.bar(
            i,
            float(avg),
            color=color,
            ecolor="k",
            yerr=std,
            label=str(i) + " - " + lbls[i]
        )
    if type == 'normal':
        #normal
        plt.xticks(range(0, len(avgs), 1))
        plt.yticks(np.arange(0,10,0.1))
        plt.xlim([-0.2, len(avgs)])
        plt.ylim([0,1])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2),
            ncol=3, fancybox=True, shadow=True)

    else:
        #corrs
        plt.xticks(range(0, len(avgs), 1))
        plt.yticks(np.arange(-10,10,0.1))
        plt.xlim([-0.2, len(avgs)])
        plt.ylim([-1,1])

    ax.set_xlabel(xLbl)
    ax.set_ylabel(yLbl)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(fname, dpi=100)

    plt.clf()
    plt.close()

def RFPers():
    lbls = ['rf_all', 'rf_eeg', 'rf_phy', 'bayesian', 'DEAP', 'top 8', 'rf_all_8', 'rf_egg_8', 'rf_phy_8']

    avgs = [0.695, 0.697, 0.689, 0.666, 0.620, 0.752, 0.794, 0.798, 0.777]
    stds = [0.074, 0.080, 0.064, 0, 0, 0, 0.042, 0.038, 0.030]
    title = 'accuracies for valence'
    genPlot(avgs, stds, lbls, title)

    avgs = [0.715, 0.693 , 0.692 , 0.664, 0.576, 0.817, 0.8381, 0.7925, 0.831]
    stds = [0.0885, 0.098, 0.0878, 0    , 0    , 0    , 0.0492, 0.023, 0.0483]
    title = 'accuracies for arousal'
    genPlot(avgs, stds, lbls, title)

def best_valence():
    clrs = ['r', 'r', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']

    # all valence
    lbls = ['bayesian','DEAP','pearsonR', 'MutInf', 'dCorr', 'LR', 'L1', 'L2', 'SVM', 'RF', 'ANOVA', 'LDA', 'PCA']
    avgs = [0.666, 0.620, 0.6875, 0.5906, 0.7 , 0.625, 0.7063, 0.625, 0.697, 0.741, 0.716, 0.619, 0.61563]
    stds = [0    , 0    , 0.13  , 0.17  , 0.13, 0.12 , 0.14  , 0.12 , 0.15 , 0.11 , 0.13 , 0.16 , 0.13   ]
    title = 'best test acc valence ALL'
    genPlot(avgs, stds, lbls, title, clrs)

    # eeg valence
    avgs = [0.666, 0.620, 0.684, 0.591, 0.713, 0.644, 0.716, 0.653, 0.697, 0.728, 0.713, 0.650, 0.606]
    stds = [0    , 0    , 0.118, 0.126, 0.122, 0.130, 0.146, 0.135, 0.151, 0.118, 0.134, 0.195, 0.134]
    title = 'best test acc valence EEG'
    genPlot(avgs, stds, lbls, title, clrs)

    # phy valence
    avgs = [0.666, 0.620, 0.600, 0.575, 0.600, 0.594, 0.581, 0.578, 0.603, 0.609, 0.566, 0.584, 0.594]
    stds = [0    , 0    , 0.141, 0.173, 0.146, 0.168, 0.147, 0.154, 0.121, 0.177, 0.185, 0.164, 0.158]
    title = 'best test acc valence PHY'
    genPlot(avgs, stds, lbls, title, clrs)

    # all vs eeg vs phy
    lbls = ['bayesian', 'DEAP','ALL', 'EEG', 'PHY']
    avgs = [0.666, 0.620, 0.741, 0.728, 0.609]
    stds = [0    , 0    , 0.109, 0.118, 0.177]
    title = 'valence ALL vs EEG vs PHY'
    genPlot(avgs, stds, lbls, title, clrs)

def best_arousal():
    clrs = ['r','r','b','b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']

    # all valence
    lbls = ['bayesian', 'DEAP','pearsonR', 'MutInf', 'dCorr', 'LR', 'L1', 'L2', 'SVM', 'RF', 'ANOVA', 'LDA', 'PCA']
    avgs = [0.664, 0.576, 0.681, 0.588, 0.713, 0.669, 0.728, 0.659, 0.688, 0.741, 0.688, 0.638, 0.597]
    stds = [0    , 0    , 0.159, 0.139, 0.127, 0.159, 0.118, 0.158, 0.132, 0.100, 0.145, 0.122, 0.145]
    title = 'best test acc arousal ALL'
    genPlot(avgs, stds, lbls, title, clrs)

    # eeg valence
    avgs = [0.664, 0.576, 0.684, 0.594, 0.716, 0.656, 0.706, 0.656, 0.669, 0.753, 0.700, 0.666, 0.581]
    stds = [0    , 0    , 0.156, 0.152, 0.123, 0.137, 0.139, 0.141, 0.136, 0.097, 0.112, 0.143, 0.149]
    title = 'best test acc arousal EEG'
    genPlot(avgs, stds, lbls, title, clrs)

    # phy valence
    avgs = [0.664, 0.576, 0.619, 0.622, 0.634, 0.594, 0.619, 0.594, 0.628, 0.650, 0.606, 0.591, 0.594]
    stds = [0    , 0    , 0.153, 0.122, 0.147, 0.141, 0.124, 0.127, 0.150, 0.127, 0.130, 0.159, 0.141]
    title = 'best test acc arousal PHY'
    genPlot(avgs, stds, lbls, title, clrs)

    # all vs eeg vs phy
    lbls = ['bayesian', 'DEAP','ALL', 'EEG', 'PHY']
    avgs = [0.664, 0.576,0.741, 0.753, 0.650]
    stds = [0,0,0.100, 0.097, 0.127]
    title = 'arousal ALL vs EEG vs PHY'
    genPlot(avgs, stds, lbls, title, clrs)

def svm_rbf_accs():
    #get testaccs
    test_accs = []

    model = PersScript("ALL", 32, 30, Classificators.ContValenceClassificator(), "../dumpedData/persScript/")
    model.run()

    for person in model.accs:
        model = person[0]

        m = []
        for metric in model:
            m.append(metric[7])
        test_accs.append(m)

    test_accs = np.array(test_accs)
    #test_accs = test_accs[:,np.array([0,1,2,3,4,5,6,7,9,10,11])]

    sort_indices = np.array([0,1,2,9, 3,6,10, 4,5,7,11])
    colors = ['b','b','b','b', 'r','r','r', 'g','g','g','g']
    names = np.array(['R', 'MI', 'dC', 'LR', 'L1', 'L2', 'SVM', 'RF', 'STD', 'ANOVA', 'LDA', 'PCA'])
    names = names[sort_indices]
    test_accs = test_accs[:, sort_indices]

    avgs = np.average(test_accs, axis=0)
    stds = np.std(test_accs, axis=0)

    genPlot(avgs,
            stds,
            names,
            'Accuracies of valence SVM models',
            'model',
            'test acc',
            colors
    )

    # get testaccs
    test_accs = []

    model = PersScript("ALL", 32, 30, Classificators.ContArousalClassificator(), "../dumpedData/persScript/")
    model.run()

    for person in model.accs:
        model = person[0]

        m = []
        for metric in model:
            m.append(metric[7])
        test_accs.append(m)

    test_accs = np.array(test_accs)
    test_accs = test_accs[:, sort_indices]
    #test_accs = test_accs[:, np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11])]

    avgs = np.average(test_accs, axis=0)
    stds = np.std(test_accs, axis=0)

    genPlot(avgs,
            stds,
            names,
            'Accuracy of arousal SVM models',
            'model',
            'test acc',
            colors
            )

    print(names)

def phyeegall():
    f = open("temp.csv", 'w')

    # get testaccs
    test_accs = []

    all = []
    model = PersScript("ALL", 32, 30, Classificators.ContValenceClassificator(),1, "D:/ir-thesis/dumpedData/persScript/")
    model.run()
    for person in model.accs:
        all.append(person[0][7][7])
    test_accs.append(all)

    all = []
    model = PersScript("EEG", 32, 30, Classificators.ContValenceClassificator(),1, "D:/ir-thesis/dumpedData/persScript/")
    model.run()
    for person in model.accs:
        all.append(person[0][7][7])
    test_accs.append(all)

    all = []
    model = PersScript("PHY", 32, 30, Classificators.ContValenceClassificator(),1, "D:/ir-thesis/dumpedData/persScript/")
    model.run()
    for person in model.accs:
        all.append(person[0][7][7])
    test_accs.append(all)

    test_accs = np.array(test_accs)

    avgs = np.average(test_accs, axis=1)
    stds = np.std(test_accs, axis=1)
    lbls = ['ALL','EEG','non-EEG']

    genPlot(avgs,
            stds,
            lbls,
            'Valence RF acc for different feat sets',
            'feature Set',
            'test acc',
            ['b','r','g']
            )
    print(avgs)
    print(stds)

    test_accs = []

    all = []
    model = PersScript("ALL", 32, 30, Classificators.ContArousalClassificator(),1,"D:/ir-thesis/dumpedData/persScript/")
    model.run()
    for person in model.accs:
        all.append(person[0][7][7])
    test_accs.append(all)

    all = []
    model = PersScript("EEG", 32, 30, Classificators.ContArousalClassificator(),1, "D:/ir-thesis/dumpedData/persScript/")
    model.run()
    for person in model.accs:
        all.append(person[0][7][7])
    test_accs.append(all)

    all = []
    model = PersScript("PHY", 32, 30, Classificators.ContArousalClassificator(),1, "D:/ir-thesis/dumpedData/persScript/")
    model.run()
    for person in model.accs:
        all.append(person[0][7][7])
    test_accs.append(all)

    test_accs = np.array(test_accs)

    avgs = np.average(test_accs, axis=1)
    stds = np.std(test_accs, axis=1)
    lbls = ['ALL', 'EEG', 'non-EEG']

    genPlot(avgs,
            stds,
            lbls,
            'Arousal RF acc for different feat sets',
            'feature Set',
            'test acc',
            ['b', 'r', 'g']
            )

    print(avgs)
    print(stds)

    f.close()


def linear_regression_example():
    X = []
    Y = []

    for i in range(100):
        x = 70 + i * 3
        y = x * 4 + random.randint(-200,200)

        X.append([x])
        Y.append(y)




    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X,Y)

    # Plot outputs
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 9)
    ax.scatter(X,Y, color='black')
    ax.plot(X, regr.predict(X), color='blue',
             linewidth=3)

    ax.set_xlabel('total area of the house (m²)')
    ax.set_ylabel('asking price ( x 1000 EUR )')
    ax.set_title("predicting the asking price of a house")

    ax.set_xticks(100 + (np.arange(7) * 50))

    fig.savefig("../results/plots/mlexample.png")
    plt.close()

    for i in range(5):
        print(str(X[i]) + ';' + str(Y[i]))

def genPiePlot(x,y,fname):
    colors = ['yellowgreen', 'red', 'gold', 'lightskyblue', 'white', 'lightcoral', 'blue', 'pink', 'darkgreen', 'yellow', 'grey', 'violet', 'magenta',
              'cyan']

    patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2)
    labels = []
    for _x,_y in zip(x,y):
        labels.append(str(_x) + ' - ' + str(_y))

    sort_legend = False
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, y),
                                             key=lambda x: x[2],
                                             reverse=True))

    #plt.legend(patches, labels, loc='lower center', bbox_to_anchor=(0, .5),
    #           fontsize=12)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 18.5)
    plt.savefig(fname, bbox_inches='tight', dpi=100)
    plt.clf()

    plt.legend(patches, x, fontsize=23, ncol=3, bbox_to_anchor=(1.,1.))
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(fname + "legend.png", dpi=100)
    plt.close()

def pieplots():
    #x = np.char.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    #y = np.array([234, 64, 54, 10, 0, 1, 0, 9, 2, 1, 7, 7])
    #genPiePlot(x,y)

    labels = ['fractions', 'power', 'asymmetry', 'heart rate', 'GSR', 'respiration', 'blood pressure', 'skin temp']

    f = open('../results/freqs.csv')
    f.readline()
    for line in f:
        line = line.strip('\n')
        (dim,featset,brol,fs,frac,power,asym,hr,gsr,rsp,bp,st) = line.split(',')

        fname = '../results/plots/' + dim+featset+fs + '.png'

        fracs = [frac,power,asym,hr,gsr,rsp,bp,st]

        genPiePlot(labels,fracs,fname)
    plt.close()
    f.close()

def corrs():
    #here no RF STD in file!!
    names = np.array(['pearson R', 'Mutual information', 'Distance correlation',
                      'Linear regression', 'Lasso regression', 'Ridge regression',
                      'SVM', 'Random forest',
                      'ANOVA', 'LDA', 'PCA']
                     )
    sort_indices = np.array([0, 1, 2, 8, 3, 6, 9, 4, 5, 7, 10])  # TODO PCA
    names = names[sort_indices]

    f = open('../results/corrs.csv')
    f.readline()
    for line in f:
        line = line.split(',')

        dim, featset = line[0], line[1]



        vals = np.array(line[2:])
        vals = vals[sort_indices]

        genPlot(vals,[0 for val in vals],names,'Correlation predict probability and level of '+ str(dim) + ' (' + str(featset) + ')','FS method','Pearson correlation', type='corrs')



    f.close()

def regions():
    f = open('../results/regions.csv')


    for line in f:
        what = line.split(',')[0]
        hdr  = f.readline().split(',')
        type = hdr[0]
        hdr = hdr[1:-1]
        vals = f.readline().split(',')[1:-1]

        fname = '../results/plots/' + what + type + ".png"

        genPiePlot(hdr, vals, fname)

    f.close()

def zones():
    labels = ['front left', 'front right', 'midline', 'back left', 'back right']
    values = [6, 6, 4, 8, 8]
    genPiePlot(labels, values, "../results/plots/arousalzones")

    values = [6,6,4,8,8]
    genPiePlot(labels, values, "../results/plots/valencezones")

    #arousal
    labels = ['front', 'back']
    values = [23, 13]
    genPiePlot(labels, values, '../results/plots/arousalasymzonesASM')

    labels = ['left','right', 'Fz/Pz']
    values = [12,19,3]
    genPiePlot(labels,values, '../results/plots/arousalasymzonesCau')

    #valence
    labels = ['front', 'back']
    values = [20, 20]
    genPiePlot(labels, values, '../results/plots/valenceasymzonesASM')

    labels = ['left', 'right', 'Fz/Pz']
    values = [11, 9, 11]
    genPiePlot(labels, values, '../results/plots/valenceasymzonesCau')

def svm_rbf_accs_gen():
    # get testaccs
    test_accs = []

    model = PersScript("ALL", 32, 30, Classificators.ContValenceClassificator(), "../dumpedData/genScript/")
    accs = load('accs_all',path=model.ddpad)[0]

    for metric in accs:
        test_accs.append(metric[7])

    test_accs = np.array(test_accs)

    sort_indices = np.array([0, 1, 2, 9, 3, 6, 10, 4, 5, 7, 11])
    colors = ['b', 'b', 'b', 'b', 'r', 'r', 'r', 'g', 'g', 'g', 'g']
    names = np.array(['R', 'MI', 'dC', 'LR', 'L1', 'L2', 'SVM', 'RF', 'STD', 'ANOVA', 'LDA', 'PCA'])
    names = names[sort_indices]
    test_accs = test_accs[sort_indices]

    genPlot(test_accs,
            [0 for acc in test_accs],
            names,
            'Accuracies of valence SVM models (cross subject)',
            'model',
            'test acc',
            colors
            )
    # get testaccs
    test_accs = []

    model = PersScript("ALL", 32, 30, Classificators.ContArousalClassificator(), "../dumpedData/genScript/")
    accs = load('accs_all', path=model.ddpad)[0]

    for metric in accs:
        test_accs.append(metric[7])

    test_accs = np.array(test_accs)

    sort_indices = np.array([0, 1, 2, 9, 3, 6, 10, 4, 5, 7, 11])
    colors = ['b', 'b', 'b', 'b', 'r', 'r', 'r', 'g', 'g', 'g', 'g']
    names = np.array(['R', 'MI', 'dC', 'LR', 'L1', 'L2', 'SVM', 'RF', 'STD', 'ANOVA', 'LDA', 'PCA'])
    names = names[sort_indices]
    test_accs = test_accs[sort_indices]

    genPlot(test_accs,
            [0 for acc in test_accs],
            names,
            'Accuracy of arousal SVM models (cross subject)',
            'model',
            'test acc',
            colors
            )

    print(names)

def phyeegall_gen():
    print('bullcrap alert! random forest not right here!!')

    lbls = ['ALL', 'EEG', 'non-EEG']

    # get testaccs
    test_accs = []
    for set in lbls:
        model = PersScript(set, 32, 30, Classificators.ContValenceClassificator(), "../dumpedData/genScript/")
        accs = load('accs_all',path=model.ddpad)[0]
        test_accs.append(accs[7][7])

    test_accs = np.array(test_accs)

    genPlot(test_accs,
            [0 for acc in test_accs],
            lbls,
            'Valence RF acc for different feat sets (cross subject)',
            'feature Set',
            'test acc',
            ['b', 'r', 'g']
            )

    print(test_accs)

    # get testaccs
    test_accs = []
    for set in lbls:
        model = PersScript(set, 32, 30, Classificators.ContArousalClassificator(), "../dumpedData/genScript/")
        accs = load('accs_all', path=model.ddpad)[0]
        test_accs.append(accs[7][7])

    test_accs = np.array(test_accs)

    genPlot(test_accs,
            [0 for acc in test_accs],
            lbls,
            'Arousal RF acc for different feat sets (cross subject)',
            'feature Set',
            'test acc',
            ['b', 'r', 'g']
            )
    print(test_accs)

def pieplotsgen():
    # x = np.char.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # y = np.array([234, 64, 54, 10, 0, 1, 0, 9, 2, 1, 7, 7])
    # genPiePlot(x,y)

    labels = ['fractions', 'power', 'asymmetry', 'heart rate', 'GSR', 'respiration', 'blood pressure', 'skin temp']

    f = open('../results/freqsgen.csv')
    f.readline()
    for line in f:
        line = line.strip('\n')
        (dim, featset, brol, fs, frac, power, asym, hr, gsr, rsp, bp, st) = line.split(',')

        fname = '../results/plots/' + dim + featset + fs + 'gen.png'

        fracs = [frac, power, asym, hr, gsr, rsp, bp, st]

        genPiePlot(labels, fracs, fname)
    plt.close()
    f.close()

def corrs_gen():
    names = np.array(['pearson R', 'Mutual information', 'Distance correlation',
                      'Linear regression', 'Lasso regression', 'Ridge regression',
                      'SVM', 'Random forest', 'STD',
                      'ANOVA', 'LDA', 'PCA']
                     )

    sort_indices = np.array([0, 1, 2, 9, 3, 6, 10, 4, 5, 7, 11])  # TODO PCA
    names = names[sort_indices]
    print(names)

    f = open('../results/corrs_gen.csv')
    f.readline()
    for line in f:
        line = line.split(';')
        print(line)
        dim, featset = line[0], line[1]

        vals = np.array(line[2:])
        vals = vals[sort_indices]

        genPlot(vals, [0 for val in vals], names, 'Correlation predict probability and level of ' + str(dim) + ' (' + str(featset) + ' )', 'FS method', 'Pearson correlation', type='corrs')

    f.close()

if __name__ == '__main__':
    phyeegall()
