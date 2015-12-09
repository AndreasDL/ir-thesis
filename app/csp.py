import numpy as np
from sklearn.preprocessing import normalize

#to preprocess data
class Csp:
    W = None

    def __init__(self, samples, labels):
        #based on
        #http://lib.ugent.be/fulltxt/RUG01/001/805/425/RUG01-001805425_2012_0001_AC.pdf
        #sampes[video][channel] = list<samples>
        
        #normalize input => avoid problems with **-.5 of P matrix
        for i in range(len(samples[0])):
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
        self.W = np.dot( np.transpose(Z), U)

        '''f = open("output filters.txt", 'w')
        for line in self.W:
            for column in line:
                f.write(str(column) + ',')
            f.write("\n")
        f.close()
        '''



    def apply(samples, channelPairs):
        #apply filters
        X_only = np.zeros((40, channelPairs * 2,8064))

        top_offset = self.channelPairs * 2 - 1
        for i, E in enumerate(samples):
            #keep the needed channels
            for j, k in zip(range(channelPairs), range(31,31-channelPairs,-1)):
                #only calculated the needed channelpairs
                X_only[i,j,:] = np.dot(self.W[j,:], E)
                X_only[i,top_offset -j,:] = np.dot(self.W[k,:], E)            

        return X_only
        
    def apply_all(self,samples):
        X_csp = []
        for i, E in enumerate(samples):
            X_csp.append(np.dot(self.W,E))

        return np.array(X_csp)