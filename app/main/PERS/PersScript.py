from scipy.stats import pearsonr, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mutual_info_score
from sklearn.cross_validation import KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt

import personLoader
import Classificators
import featureExtractor as FE
from personLoader import load, dump
import numpy as np

from multiprocessing import Pool
POOL_SIZE = 2


class PersScript():
    def __init__(self, feats, stopperson, threshold, classifier, ddpad = "../../../dumpedData/persScript/"):
        self.featExtr = FE.MultiFeatureExtractor()
        self.feats = feats
        if feats == 'EEG':
            self.addEEGFeatures()
        elif feats == 'PHY':
            self.addPhyFeatures()
        else:
            self.addEEGFeatures()
            self.addPhyFeatures()

        self.stopperson = stopperson
        self.threshold = threshold

        self.classifier = classifier

        self.ddpad = ddpad
        if self.classifier.name == "ContArousalClasses":
            self.ddpad += "arousal/"
        else:
            self.ddpad += "valence/"

        if feats == "EEG":
            self.ddpad += "eeg/"
        elif feats == "PHY":
            self.ddpad += "phy/"
        else:
            self.ddpad += "all/"

        self.rpad = self.ddpad + "results/"

        self.metricnames =  ['pearsonR','MutInf','dCorr','LR','L1','L2','SVM','RF', 'RFSTD', 'ANOVA','LDA', 'PCA']
        self.modelnames  = ["SVMLIN","SVMRBF","KNN3","KNN5","KNN7"]

    def addEEGFeatures(self):

        # EEG
        for channel in FE.all_EEG_channels:

            self.featExtr.addFE(
                FE.AlphaBetaExtractor(
                    channels=[channel],
                    featName='A/B ' + FE.all_channels[channel]
                )
            )

            for freqband in FE.startFreq:

                if freqband != 'all':
                    self.featExtr.addFE(
                        FE.BandFracExtractor(
                            channels=[channel],
                            featName='frac' + str(FE.all_channels[channel]) + "-" + str(freqband),
                            freqBand=freqband
                        )
                    )

                self.featExtr.addFE(
                    FE.DEExtractor(
                        channels=[channel],
                        freqBand=freqband,
                        featName='DE ' + FE.all_channels[channel] + '(' + freqband + ')'
                    )
                )

                self.featExtr.addFE(
                    FE.PSDExtractor(
                        channels=[channel],
                        freqBand=freqband,
                        featName='PSD ' + FE.all_channels[channel] + '(' + freqband + ')'
                    )
                )

        for left, right in zip(FE.all_left_channels, FE.all_right_channels):

            for freqband in FE.startFreq:
                self.featExtr.addFE(
                    FE.DASMExtractor(
                        left_channels=[left],
                        right_channels=[right],
                        freqBand=freqband,
                        featName='DASM ' + FE.all_channels[left] + ',' + FE.all_channels[right]
                    )
                )

                self.featExtr.addFE(
                    FE.RASMExtractor(
                        left_channels=[left],
                        right_channels=[right],
                        freqBand=freqband,
                        featName='RASM ' + FE.all_channels[left] + ',' + FE.all_channels[right]
                    )
                )

        for front, post in zip(FE.all_frontal_channels, FE.all_posterior_channels):
            for freqband in FE.startFreq:
                self.featExtr.addFE(
                    FE.DCAUExtractor(
                        frontal_channels=[front],
                        posterior_channels=[post],
                        freqBand=freqband,
                        featName='DCAU ' + FE.all_channels[front] + ',' + FE.all_channels[post]
                    )
                )

                self.featExtr.addFE(
                    FE.RCAUExtractor(
                        frontal_channels=[front],
                        posterior_channels=[post],
                        freqBand=freqband,
                        featName='RCAU ' + FE.all_channels[front] + ',' + FE.all_channels[post]
                    )
                )
    def addPhyFeatures(self):
        # physiological signals
        for channel in FE.all_phy_channels:
            self.featExtr.addFE(FE.AvgExtractor(channel, ''))
            self.featExtr.addFE(FE.STDExtractor(channel, ''))
            self.featExtr.addFE(FE.MaxExtractor(channel, ''))
            self.featExtr.addFE(FE.MinExtractor(channel, ''))
            self.featExtr.addFE(FE.MedianExtractor(channel, ''))
            self.featExtr.addFE(FE.VarExtractor(channel, ''))

        self.featExtr.addFE(FE.AVGHeartRateExtractor())
        self.featExtr.addFE(FE.STDInterBeatExtractor())
        self.featExtr.addFE(FE.MaxHRExtractor())
        self.featExtr.addFE(FE.MinHRExtractor())
        self.featExtr.addFE(FE.MedianHRExtractor())
        self.featExtr.addFE(FE.VarHRExtractor())

    def d_n(self, x):
        d = np.abs(x[:, None] - x)
        dn = d - d.mean(0) - d.mean(1)[:, None] + d.mean()
        return dn
    def dcov_all(self, x, y):
        # stolen with pride from https://gist.githubusercontent.com/josef-pkt/2938402/raw/ff949d6379484e9fc745bd534bdfc5d22ad074e2/try_distance_corr.py
        dnx = self.d_n(x)
        dny = self.d_n(y)

        denom = np.product(dnx.shape)
        dc = (dnx * dny).sum() / denom
        dvx = (dnx ** 2).sum() / denom
        dvy = (dny ** 2).sum() / denom
        dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
        return np.sqrt(dc), np.sqrt(dr), np.sqrt(dvx), np.sqrt(dvy)
    def runPers(self,person):
        #load person data
        pers_results = load('pers_res_p' + str(person), path=self.ddpad)
        if pers_results == None:
            pers_results = []

            # load all features & keep them in memory
            y_cont = load('cont_y_p' + str(person), path=self.ddpad)
            if y_cont == None:
                print('[Warn] Rebuilding cache -  person ' + str(person))
                personLdr = personLoader.NoTestsetLoader(self.classifier, self.featExtr)

                X, y_cont = personLdr.load(person)

                dump(X, 'X_p' + str(person), path=self.ddpad)
                dump(y_cont, 'cont_y_p' + str(person), path=self.ddpad)
            else:
                X = load('X_p' + str(person), path=self.ddpad)

            y_disc = np.array(y_cont)
            y_disc[y_disc <= 5] = 0
            y_disc[y_disc > 5] = 1

            # manual Feature standardization
            X = X - np.average(X, axis=0)
            X = np.true_divide(X, np.std(X, axis=0))

            #pearson
            corr = []
            for index in range(len(X[0])):
                corr.append( pearsonr(X[:, index], y_cont)[0] )
            pers_results.append(corr)

            #Mut inf
            #dcorr
            mi = []
            dcorr = []
            for feature in np.transpose(X):
                # normalized mutual information
                c_xy = np.histogram2d(feature, y_cont, 2)[0]
                entX = entropy(feature, y_cont)
                entY = entropy(feature, y_cont)
                nMutInf = mutual_info_score(None, None, contingency=c_xy) / float(np.sqrt(entX * entY))
                mi.append(nMutInf)

                # Distance Correlation
                dc, dr, dvx, dvy = self.dcov_all(feature, y_cont)
                dcorr.append(dr)

            pers_results.append(mi)
            pers_results.append(dcorr)

            #Linear Regression
            lr = LinearRegression(n_jobs=-1)
            lr.fit(X, y_cont)
            pers_results.append(lr.coef_)

            #Lasso Regression
            alphas = [0.03, 0.1, 0.3, 1, 3, 10]
            best_alpha = 0.01
            best_acc = 0
            for train_index, test_index in KFold(len(y_cont), n_folds=5):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_cont[train_index], y_cont[test_index]

                lasso = Lasso(alpha=best_alpha)
                lasso.fit(X_train, y_train)
                pred = lasso.predict(X_test)
                best_acc += self.accuracy(pred, y_cont)
            best_acc /= float(5)

            for alpha in alphas:
                acc = 0
                for train_index, test_index in KFold(len(y_cont), n_folds=5):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y_cont[train_index], y_cont[test_index]

                    lasso = Lasso(alpha=alpha)
                    lasso.fit(X_train, y_train)
                    pred = lasso.predict(X_test)
                    acc += self.accuracy(pred, y_test)

                acc /= float(5)
                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha

            lasso = Lasso(alpha=best_alpha)
            lasso.fit(X, y_cont)
            pers_results.append(lasso.coef_)

            #Ridge Regression
            alphas = [0.03, 0.1, 0.3, 1, 3, 10]
            best_alpha = 0.01
            best_acc = 0
            for train_index, test_index in KFold(len(y_cont), n_folds=5):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_cont[train_index], y_cont[test_index]

                ridge = Ridge(alpha=best_alpha)
                ridge.fit(X_train, y_train)
                pred = ridge.predict(X_test)
                best_acc += self.accuracy(pred, y_test)
            best_acc /= float(5)

            for alpha in alphas:
                acc = 0
                for train_index, test_index in KFold(len(y_cont), n_folds=5):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y_cont[train_index], y_cont[test_index]

                    ridge = Ridge(alpha=alpha)
                    ridge.fit(X_train, y_train)
                    pred = lasso.predict(X_test)
                    acc += self.accuracy(pred, y_test)

                acc /= float(5)
                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha

            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X, y_cont)
            pers_results.append(ridge.coef_)

            #SVM
            clf = SVC(kernel='linear')
            clf.fit(X, y_disc)
            svm_weights = (clf.coef_ ** 2).sum(axis=0)
            svm_weights /= float(svm_weights.max())
            pers_results.append(svm_weights)

            #Random Forests
            #rf importances
            #grow forest
            forest = RandomForestClassifier(
                n_estimators=2000,
                max_features='auto',
                criterion='gini',
                n_jobs=-1,
            )
            forest.fit(X,y_disc)
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
            pers_results.append(importances)
            pers_results.append(std)

            #ANOVA
            anova = SelectKBest(f_regression, k=self.threshold)
            anova.fit(X,y_disc)
            selected_features = anova.get_support()
            pers_results.append(selected_features)

            #Linear Discriminant Analysis
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(X,y_disc)
            pers_results.append(lda.coef_[0])

            #Principal Component Analysis
            pca = PCA(n_components=1)
            pca.fit(X)
            pers_results.append(pca.components_[0])

            #absolute values
            pers_results = np.absolute(np.array(pers_results))

            dump(pers_results, 'pers_res_p' + str(person), path=self.ddpad)

        return np.array(pers_results)
    def genReport(self):

        #self.results[person][metric][feature] = value
        for person, persData in enumerate(self.results):
            f = open(self.rpad + "results" + str(person) + ".csv", 'w')
            f.write('Person: ' + str(person) + '\nfeatName;pearsonR;MutInf;dCorr;LR;L1;L2;SVM;RF;ANOVA;LDA;\n')

            for index,featName in zip(range(len(persData[0])),self.featExtr.getFeatureNames()): #all metrics
                f.write(featName + ";")
                for value in persData[:,index]:
                    f.write(str(value) +  ';')
                f.write('\n')
            f.write('\n\n\n')

            f.close()

    def accuracy(self, predictions, truths):
        acc = 0
        for pred, truth in zip(predictions, truths):
            acc += (pred == truth)

        return acc / float(len(predictions))
    def getAccs(self, person):
        print('gettings accs for person ' + str(person))

        to_ret = load('accs_p' + str(person), path = self.ddpad)
        if to_ret == None:

            # load person data
            # load all features & keep them in memory
            y = load('cont_y_p' + str(person), path=self.ddpad)
            if y == None:
                print('[Warn] Rebuilding cache -  person ' + str(person))
                personLdr = personLoader.NoTestsetLoader(self.classifier, self.featExtr)

                X, y = personLdr.load(person)

                dump(X, 'X_p' + str(person), path=self.ddpad)
                dump(y, 'cont_y_p' + str(person), path=self.ddpad)
            else:
                X = load('X_p' + str(person), path=self.ddpad)

            y = np.array(y)


            # manual Feature standardization
            X = X - np.average(X, axis=0)
            X = np.true_divide(X, np.std(X, axis=0))

            #train test set
            # Train / testset
            X, X_test, y, y_test_cont = train_test_split(X,y,test_size=10, random_state=17)

            y[y <= 5] = 0
            y[y >  5] = 1

            y_test = np.array(y_test_cont)
            y_test[y_test <= 5] = 0
            y_test[y_test > 5] = 1

            to_ret = []

            for model_index, model in enumerate([
                #SVC(kernel='linear'),
                SVC(kernel='rbf'),

                #KNN(n_neighbors=3),
                #KNN(n_neighbors=5),
                #KNN(n_neighbors=7),

                #lda needs two features

                RandomForestClassifier(
                    n_estimators=1000,
                    max_features='auto',
                    criterion='gini',
                    n_jobs=-1,
                )
            ]):
                model_to_ret = []
                for metric in self.results[person-1]:
                    featNames = np.array(self.featExtr.getFeatureNames()) #take clean copy

                    #sort features
                    indices = np.array(np.argsort(metric)[::-1])
                    #take top threshold
                    indices = indices[:self.threshold]

                    #apply
                    X_model = np.array(X[:,indices])
                    X_model_test = np.array(X_test[:,indices])
                    featNames = featNames[indices]

                    best_feat, best_featNames = [], []
                    all_scores, all_stds = [],[]
                    best_score, best_std = 0, 0
                    for i in range(self.threshold):
                        to_keep = best_feat[:]
                        to_keep.append(i)

                        X_temp = np.array(X_model[:,to_keep])

                        # get scores
                        run_scores = []
                        for tr, te in KFold(n=len(X_temp), n_folds=5, shuffle=True, random_state=17):
                            model.fit(X_temp[tr], y[tr])
                            run_scores.append(self.accuracy(model.predict(X_temp[te]), y[te]))

                        new_score = np.average(run_scores)
                        new_std = np.std(run_scores)

                        all_scores.append(new_score)
                        all_stds.append(new_std)

                        # better?
                        if new_score - new_std > best_score - best_std:
                            best_score = new_score
                            best_std = new_std
                            best_feat = to_keep
                            best_featNames.append(featNames[i])

                    #get test score
                    if model_index == 0:
                        test_model = SVC(kernel='rbf', probability=True)
                        test_model.fit(X_model[:,best_feat], y)

                        X_model_test = np.array(X_model_test[:,best_feat])
                        test_pred = test_model.predict(X_model_test)
                        test_prob = test_model.predict_proba(X_model_test)

                        test_acc = self.accuracy(test_model.predict(X_model_test), y_test)
                    else:
                        test_model = RandomForestClassifier(
                            n_estimators=2000,
                            max_features='auto',
                            criterion='gini',
                            n_jobs=-1
                        )
                        test_model.fit(X_model[:, best_feat], y)

                        X_model_test = np.array(X_model_test[:, best_feat])
                        test_pred = test_model.predict(X_model_test)
                        test_prob = test_model.predict_proba(X_model_test)

                        test_acc = self.accuracy(test_model.predict(X_model_test), y_test)

                    model_to_ret.append([best_feat, best_featNames, best_score, best_std, all_scores, all_stds, indices, test_acc, test_pred, test_prob, y_test, y_test_cont])
                to_ret.append(model_to_ret)

            dump(to_ret, 'accs_p' + str(person), path = self.ddpad)

        return to_ret
    def genAccReport(self):
        #self.accs[person][model][metric] = [best_feat, best_featNames, best_score, best_std, all_score, all_std]

        tekst = []
        for person in range(len(self.accs)):
            f = open(self.rpad + "Accresults" + str(person) + ".csv", 'w')

            #accuracies
            f.write("model;pearsonR;MutInf;dCorr;LR;L1;L2;SVM;RF;ANOVA;LDA;\n")
            for model, modelName in zip(self.accs[person], ["SVMLIN","SVMRBF","KNN3","KNN5","KNN7"]):
                f.write(modelName + ';')

                t = []
                for metric in model:
                    t.append([])
                    f.write(str(metric[2]) + '(' + str(metric[3]) + ');')
                f.write('\n')

                if person == 0:
                    tekst.append(t)


            f.write('\n\n\n')

            #features
            for modelIndex, (model, modelName) in enumerate(zip(self.accs[person], ["SVMLIN","SVMRBF","KNN3","KNN5","KNN7"])):
                f.write(modelName + '\n\n')
                f.write("matric;features used;\n")
                for metricIndex, (metric, metricName) in enumerate(zip(model, ['pearsonR','MutInf','dCorr','LR','L1','L2','SVM','RF','ANOVA','LDA'])):
                    f.write(metricName + ';')
                    for featName in metric[1]:
                        f.write(featName + ';')
                    f.write('\n')

                    tekst[modelIndex][metricIndex].append(metric[1])
                f.write('\n\n\n')

            f.close()

        for model, modelName in zip(tekst, ["SVMLIN","SVMRBF","KNN3","KNN5","KNN7"]):
            g = open(self.rpad + "finFeat" + str(modelName) + ".csv", 'w')

            for metric, metricName in zip(model, ['pearsonR','MutInf','dCorr','LR','L1','L2','SVM','RF','ANOVA','LDA']):
                g.write(metricName + "\n")
                g.write('person;usedFeatures;\n')

                for person,data in enumerate(metric):
                    g.write(str(person) + ';')
                    for feat in data:
                        g.write(str(feat) + ';')
                    g.write('\n')

                g.write("\n\n\n'")
            g.close()

    def genFinalReport(self):
        avgs, stds = [], []
        for metric in range(12):
            avg_metric_results = []
            std_metric_results = []
            for model in range(5):
                model_results = []
                for person in range(32):
                    model_results.append(self.accs[person][model][metric][7])
                avg_metric_results.append(np.average(model_results))
                std_metric_results.append(np.std(model_results))
            avgs.append(avg_metric_results)
            stds.append(std_metric_results)



        tail = ''
        if self.classifier.name == "ContArousalClasses":
            tail += "arousal_"
        else:
            tail += "valence_"

        if self.feats == "EEG":
            tail += "eeg"
        elif self.feats == "PHY":
            tail += "phy"
        else:
            tail += "all"

        f = open(self.rpad + "finalReport_" + tail + '.csv', 'w')


        #header line
        f.write("metricName;")
        for name in self.metricnames:
            f.write(name + ';;')
        f.write('\n')

        for metIndex, (metricname, avg_metric, std_metric) in enumerate(zip(self.metricnames,avgs, stds)):
            if metIndex == 8: #RF STD
                continue
            else:
                #create acc overview
                f.write(metricname + ';')
                for avg_model, std_model in zip(avg_metric, std_metric):
                    f.write(str(round(avg_model,5)) + ';' + str(round(std_model,5)) + ';')
                f.write('\n')

                #create plot
                self.genPlot(avg_metric,
                             std_metric,
                             self.modelnames,
                             tail + metricname
                             )

        f.close()

        g = open(self.rpad + "lastIndex" + tail + '.csv', 'w')
        g.write("metricName;")
        for name in self.modelnames:
            g.write(name + ';')
        g.write('\n')

        # indices of last selected feat index
        avg_lasts = []
        std_lasts = []
        for metric in range(12):
            avg_metric_last = []
            std_metric_last = []
            for model in range(5):
                lastIndexes = []
                for person in range(32):
                    lastIndexes.append(self.accs[person][model][metric][0][-1])
                avg_metric_last.append(np.average(lastIndexes))
                std_metric_last.append(np.std(lastIndexes))
            avg_lasts.append(avg_metric_last)
            std_lasts.append(std_metric_last)


        for metIndex, (metricname, avg_last, std_last) in enumerate(zip(self.metricnames, avg_lasts, std_lasts)):
            if metIndex == 8:  # RF STD
                continue
            else:
                # create acc overview
                g.write(metricname + ';')
                for l, s in zip(avg_last, std_last):
                    g.write(str(l) + ' (' + str(s) + ')' + ';')
                g.write('\n')

        g.close()


        featnames = np.array(self.featExtr.getFeatureNames())

        for person, personData in enumerate(self.accs):
            h = open(self.rpad + "selectedFeat" + tail + '_' + str(person) + '.csv', 'w')

            model =personData[0]
            for metricName, metric in zip(self.metricnames,model):
                h.write(metricName + ';')

                feats = featnames[metric[6]]
                for feat in feats:#sorted(feats):
                    h.write(feat + ';')

                h.write('\n')
        h.close()


        h = open(self.rpad + "test_accs.csv",'w')
        for person, personData in enumerate(self.accs):
            model =personData[1]

            h.write(str(person) + ';')
            for metric in model:
                h.write(str(metric[7]) + ';')
            h.write('\n')

        h.close()

    def probVsDim(self):


        data = [
            [
                [] for i in range(len(self.accs[0][0]))
            ] for j in range(len(self.accs[0]))
        ]

        for person_index, person in enumerate(self.accs):
            for model_index, model in enumerate(person):
                for metric_index, metric in enumerate(model):
                    #[best_feat, best_featNames, best_score, best_std, all_scores, all_stds, indices, test_acc, test_pred, test_prob, y_test, y_test_cont]
                    for (pred_lbl, probs, test_lbl, value) in zip(metric[-4], metric[-3], metric[-2], metric[-1]):
                        if pred_lbl == test_lbl:
                            data[model_index][metric_index].append((
                                probs[pred_lbl],
                                np.abs(value - 5)
                                ))

        dump(data, "corrData", path=self.ddpad)

        f = open(self.rpad + "corrs.csv", 'w')

        # header line
        f.write('model;')
        for name in self.metricnames:
            f.write(name + ';')
        f.write('\n')

        #much efficiency wow </sarcasm>
        modelNames = ['SVM','RF']
        for model_index, model in enumerate(data):
            f.write(modelNames[model_index] + ';')

            for metric in model:
                probs = []
                dists = []
                for tup in metric:
                    probs.append(tup[0])
                    dists.append(tup[1])

                corr = pearsonr(dists, probs)[0]
                f.write(str(corr) + ';')

            f.write('\n')
        f.close()

    def genPlot(self,avgs, stds, lbls, title):

        fname = self.rpad + 'accComp_' + str(title) + '.png'

        # Plot the feature importances of the forest
        fig, ax = plt.subplots()

        plt.title(title)
        for i, (avg, std) in enumerate(zip(avgs, stds)):
            color = ""
            if i < 2:
                color = "r"
            else:
                color = "b"

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
        plt.clf()
        plt.close()

    def run(self):

        pool = Pool(processes=POOL_SIZE)
        self.results = pool.map(self.runPers, range(1, self.stopperson + 1))
        pool.close()
        pool.join()

        self.genReport()

        pool = Pool(processes=POOL_SIZE)
        self.accs = pool.map(self.getAccs, range(1, self.stopperson + 1))
        pool.close()
        pool.join()

        self.genAccReport()

        self.probVsDim()
        #self.genFinalReport()

if __name__ == '__main__':
    PersScript("PHY", 32, 30, Classificators.ContValenceClassificator()).run()
    PersScript("EEG", 32, 30, Classificators.ContValenceClassificator()).run()
    PersScript("ALL", 32, 30, Classificators.ContValenceClassificator()).run()

    PersScript("EEG", 32, 30, Classificators.ContArousalClassificator()).run()
    PersScript("PHY", 32, 30, Classificators.ContArousalClassificator()).run()
    PersScript("ALL", 32, 30, Classificators.ContArousalClassificator()).run()

