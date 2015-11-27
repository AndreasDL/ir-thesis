import os
import numpy as np

from numpy import linalg as LA
from sklearn.preprocessing import normalize


def calc_filters(train, train_labels):
    train = np.array(train)

    #make avg value equal to zero
    avg_channel = np.zeros(32)
    for i in range(32):
        avg_channel[i] += np.sum(train[:,i,:])
    avg_channel /= float(8064 * 40)

    for i in range(32):
        train[:,i,:] -= avg_channel[i]


    cov1 = 0
    cov2 = 0
    for j in range(len(train)):
        E = np.transpose(train[j])

        prod = np.dot(train[j], E)

        if train_labels[j] == 0:
            cov1 += prod/np.trace(prod)
        elif train_labels[j] == 1:
            cov2 += prod/np.trace(prod)

    cov = cov1 + cov2
    
    #   Covariantiematrix diagonaliseren
    p, v = LA.eig(cov)
    #p = LA.eig(cov)[0];
    p = np.diag(p)
    
    #   Whitening-transformatie op de gediagonaliseerde covariantiematrix
    U = np.dot(np.sqrt(LA.inv(p)), np.transpose(v))

    R1 = np.dot(U, np.dot(cov1, np.transpose(U)))
    #R2 = np.dot(U, np.dot(cov2, np.transpose(U)))

    #   R1 Diagonaliseren
    d, z = LA.eig(R1)
    #d = LA.eig(R1)[0]

    #   De eigenwaarden moeten gesorteerd zijn, zodat we snel zien welke filters het best presteren
    ind = d.ravel().argsort()
    d = np.sort(d)
    z = z[:, ind]

    d = np.diag(d)

    W = np.dot(np.transpose(z), U)
                      
    return d, W


def selectN(aantal_filterparen, W, d, trials_train, trials_train_labels):
    # trials_train = 160 * 800 * 60

    N = aantal_filterparen*2

    design_matrix = np.zeros((len(trials_train),N))     #160*N
    label_matrix = np.zeros((len(trials_train),1))  

    for j in range(len(trials_train)):
        E = trials_train[j] # 60 * 800
        E_csp = np.dot(W,E)
    
        # Kenmerken van elke kalibratietrial berekenen volgens fi = log(VAR(ei_CSP)) i=1..N
        # Dit geeft een 160xN design matrix
        # Een 160x1 label matrix

        for i in range(int(N/2)):
            design_matrix[j,i] = np.log(np.var(E_csp[i,:]))
            design_matrix[j,N-1-i] = np.log(np.var(E_csp[len(d)-(i+1),:]))

        # Dit is eigenlijk overbodig, aangezien de labels niet veranderen
        label_matrix[j]=trials_train_labels[j]

    # features normaliseren
    design_matrix = normalize(design_matrix)

    return design_matrix, label_matrix


def applyN(aantal_filterparen, W, d, trials_test):
    N = aantal_filterparen*2
    design_matrix_test = np.zeros((len(trials_test),N))

    for j in range(len(trials_test)):
        E_test = np.transpose(trials_test[j])
        E_csp_test = np.dot(W,E_test)
        
        for i in range (N/2):
            design_matrix_test[j,i] = np.log(np.var(E_csp_test[i,:]))
            design_matrix_test[j,N-1-i] = np.log(np.var(E_csp_test[len(d)-(i+1),:]))

    # features normaliseren
    design_matrix_test = normalize(design_matrix_test)

    return design_matrix_test


