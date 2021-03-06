from sklearn.metrics import jaccard_similarity_score
import numpy as np

ddpath = '../dumpedData/'

trees = 2000
for setting in ['pers','gen']:
    print(setting)

    for dim in ['valence','arousal']:
        for featSet in ['all']:
            for runs in [1,5,10,20,30,40,50]:
                #open first & second trial
                ppath  = ddpath
                ppath += setting
                ppath += 'Script_run0_' + str(trees) + '_' + str(runs) + '/'
                ppath += dim + '/'
                ppath += featSet + '/results/finFeatSVM'
                if setting == 'pers':
                    ppath += 'LIN'
                else:
                    ppath += 'RBF'
                ppath += '.csv'

                npath  = ddpath
                npath += setting
                npath += 'Script_run1_' + str(trees) + '_' + str(runs) + '/'
                npath += dim + '/'
                npath += featSet + '/results/finFeatSVM'
                if setting == 'pers':
                    npath += 'LIN'
                else:
                    npath += 'RBF'
                npath += '.csv'

                pfile = open(ppath, 'r')
                nfile = open(npath, 'r')

                jaccards = []
                stopIndex = 34
                if setting == 'gen':
                    stopIndex = 3
                for person, (pline, nline) in enumerate(zip(pfile,nfile)):

                    if person > 1 and person < stopIndex:
                        #print(pline.strip('\n'))

                        pfeats = pline.strip('\n').split(';')
                        nfeats = nline.strip('\n').split(';')

                        #fix length issues
                        if len(pfeats) < len(nfeats):
                            diff = len(nfeats) - len(pfeats)
                            for i in range(diff):
                                pfeats.append('')
                        elif len(nfeats) < len(pfeats):
                            diff = len(pfeats) - len(nfeats)
                            for i in range(diff):
                                nfeats.append('')

                        #jaccard
                        jackie = jaccard_similarity_score(pfeats, nfeats)

                        #prints
                        #print('\t' + pline, end='')
                        #print('\t' + nline, end='')
                        #print('\t' + str(jackie))

                        jaccards.append(jackie)

                pfile.close()
                nfile.close()

                print("\t" + dim + ';' + featSet + ';' + str(runs) + ';' + str(np.average(jaccards)) + ';' + str(np.std(jaccards)))
