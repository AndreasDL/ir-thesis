import matplotlib.pylab as plt
import numpy as np


def genPlot(avgs,stds,lbls,title, fpad="../results/plots/"):

    fname = fpad + 'accComp_' + str(title) + '.png'

    # Plot the feature importances of the forest
    fig, ax = plt.subplots()

    plt.title(title)
    for i, (avg, std) in enumerate(zip(avgs,stds)):
        color = ""
        if i < 3:
            color = "r"
        elif i < 6:
            color = "b"
        else:
            color = "g"

        ax.bar(
            i,
            avg,
            color=color,
            ecolor="k",
            yerr=std,
            label=lbls[i]
        )

    plt.xticks(range(0, len(avgs), 1))
    plt.xlim([-0.2, len(avgs)])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.25),
              ncol=3, fancybox=True, shadow=True)

    plt.savefig(fname)

    plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    lbls = ['rf_all', 'rf_eeg', 'rf_phy', 'bayesian', 'DEAP', 'top 8', 'rf_all_8', 'rf_egg_8', 'rf_phy_8']


    avgs = [0.695,0.697,0.689,0.666,0.620,0.752,0.794,0.798,0.777]
    stds = [0.074,0.080,0.064,0,0,0,0.042,0.038,0.030]
    title = 'accuracies for valence'
    genPlot(avgs,stds,lbls,title)

    avgs = [0.715,0.693,0.692,0.664,0.576,0.817,0.8381,0.7925,0.831]
    stds = [0.0885,0.0981,0.0878,0,0,0,0.0492,0.023,0.0483]
    title = 'accuracies for arousal'
    genPlot(avgs,stds,lbls,title)