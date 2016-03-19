from personLoader import load
import reporters
import numpy as np

importances = np.array(load('importances_once'))
feat_names = [ importances[0,i,0].featureName for i in range(len(importances[0,:]))]
avg_importances = np.average(importances[:,:,1], axis=0)
std_importances = [ np.std(importances[:,i,1]) for i in range(len(importances[0]))]

reporters.GenericReporter('RF_valence').genReport({'feat_names': feat_names ,
    'avg_weights' : avg_importances,
    'std_weights' : std_importances,
})