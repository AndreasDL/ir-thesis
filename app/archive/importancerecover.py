from personLoader import load
import reporters
import numpy as np
import datetime
import time
import types
import featureExtractor
from featureExtractor import all_channels




what_mapper = {
    featureExtractor.PSDExtractor: 'PSD',
    featureExtractor.DEExtractor: 'DE',
    featureExtractor.RASMExtractor: 'RASM',
    featureExtractor.DASMExtractor: 'DASM',
    featureExtractor.AlphaBetaExtractor: 'AB',
    featureExtractor.DCAUExtractor: 'DCAU',
    featureExtractor.RCAUExtractor: 'RCAU',
    featureExtractor.LMinRLPlusRExtractor: 'LminR',
    featureExtractor.FrontalMidlinePower: 'FM',
    featureExtractor.AvgExtractor: 'AVG',
    featureExtractor.STDExtractor: 'STD',
    featureExtractor.AVGHeartRateExtractor: 'AVG HR',
    featureExtractor.STDInterBeatExtractor: 'STD HR'
}

feat_names = []
feat_eeg = []
feat_what = []
feat_channels = []
feat_wavebands = []

importances = np.array(load('importances_once'))
for featExtr in importances[0,:,0]:

    feat_names.append(featExtr.featureName)

    if type(featExtr) in [featureExtractor.AVGHeartRateExtractor, featureExtractor.AvgExtractor, featureExtractor.STDExtractor, featureExtractor.STDInterBeatExtractor]:
        feat_eeg.append('phy')

        #no freq bands
        feat_wavebands.append('n/a')
        feat_channels.append('n/a')

    else:
        feat_eeg.append('eeg')

        #single channel
        if type(featExtr) in [featureExtractor.PSDExtractor, featureExtractor.DEExtractor, featureExtractor.FrontalMidlinePower]:

            feat_wavebands.append(featExtr.usedFeqBand)
            feat_channels.append(all_channels[featExtr.usedChannelIndexes[0]])

        elif type(featExtr) == featureExtractor.AlphaBetaExtractor:
            feat_wavebands.append('alpha & beta')
            feat_channels.append(all_channels[featExtr.usedChannelIndexes[0]])

        elif type(featExtr) == featureExtractor.LMinRLPlusRExtractor:
            feat_wavebands.append('alpha')
            feat_channels.append( [
                all_channels[featExtr.left_channels[0]],
                all_channels[featExtr.right_channels[0]]
            ])


        #multiple channels Left and right
        elif type(featExtr) in [featureExtractor.DASMExtractor, featureExtractor.RASMExtractor]:

            feat_wavebands.append(featExtr.leftDEExtractor.usedFeqBand)

            feat_channels.append( [
                all_channels[featExtr.leftDEExtractor.usedChannelIndexes[0]],
                all_channels[featExtr.rightDEExtractor.usedChannelIndexes[0]]
            ])

        #multiple channels post and front
        else:
            feat_wavebands.append(featExtr.frontalDEExtractor.usedFeqBand)

            feat_channels.append([
                all_channels[featExtr.frontalDEExtractor.usedChannelIndexes[0]],
                all_channels[featExtr.posteriorDEExtractor.usedChannelIndexes[0]]
            ])


    feat_what.append(what_mapper[type(featExtr)])


avg_importances = np.average(importances[:,:,1], axis=0)
std_importances = [ np.std(importances[:,i,1]) for i in range(len(importances[0]))]


#output to file
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
f = open('../../results/RF_valence' + str(st) + ".csv", 'w')

f.write('featname;eeg/phy;what;channels;waveband;avg weight;std weight;\n')

for name,eeg,what,channels,wavebands,avg,std in zip(feat_names,feat_eeg, feat_what, feat_channels, feat_wavebands, avg_importances, std_importances):
    f.write(
        str(name) + ';' +
        str(eeg) + ';' +
        str(what) + ';' +
        str(channels) + ';' +
        str(wavebands) + ';' +
        str(avg) + ';' +
        str(std) + '\n'
    )

f.close()