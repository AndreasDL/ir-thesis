import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize


def csp(samples, labels, channelPairs=1):
    #based on
    #http://lib.ugent.be/fulltxt/RUG01/001/805/425/RUG01-001805425_2012_0001_AC.pdf
    #sampes[video][channel] = list<samples>
    
    #normalize input => avoid problems with **-.5 of P matrix
    for i in range(32):
        samples[:,i,:] = normalize(samples[:,i,:])

    #divide in two classes
    cov0 = 0
    cov1 = 0
    for klass, sample in zip(labels, samples):
        #each sample = 32 channels x 8064 samples
        #avg = nul?
        E = sample
        Et = np.transpose(sample)
        
        prod = np.dot(E, Et)

        if klass == 0:
            cov0 += prod/np.trace(prod)
        else:
            cov1 += prod/np.trace(prod)
        
    #step one
    cov = cov0 + cov1

    #step two Vt * E * V = P
    P, V = np.linalg.eig(cov)
    P = np.diag(P)
    
    #step three whitening
    U = np.dot( np.sqrt(np.linalg.inv(P)), np.transpose(V) )
    Rnul = np.dot( U, np.dot(cov0, np.transpose(U)) )
    #Reen = np.dot( U, np.dot(cov1, np.transpose(U)) ) #is not used => don't calculate

    #step four Zt * R0 * Z = D
    D, Z = np.linalg.eig(Rnul)
    
    #sorteer eigenwaarden op diagonaal
    idx = D.ravel().argsort()
    Z = Z[:,idx]
    D = D[idx]
    #put eigen values on diagonal
    D = np.diag(D)
    
    #step five
    W = np.dot( np.transpose(Z), U)
    #print(D)
    #print(W)

    #apply filters
    X_only = np.zeros((40,channelPairs * 2,8064))

    top_offset = channelPairs * 2 - 1
    for i, E in enumerate(samples):
        #keep the needed channels
        for j, k in zip(range(channelPairs), range(31,31-channelPairs,-1)):
            #only calculated the needed channelpairs
            X_only[i,j,:] = np.dot(W[j,:], E)
            X_only[i,top_offset -j,:] = np.dot(W[k,:], E)            

    return X_only, W

def accuracy(predictions, truths):
    acc = 0
    for pred, truth in zip(predictions, truths):
        acc += (pred == truth)

    return acc / float(len(predictions))

def tptnfpfn(predictions, truths):
    tp, tn = 0, 0
    fp, fn = 0, 0

    for pred, truth in zip(predictions, truths):
        if pred == truth: #prediction is true
            if pred == 1:
                tp += 1
            else:
                tn += 1
        
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
        
    return tp, tn, fp, fn

def auc(predictions, truths):

    return roc_auc_score(truths, predictions)