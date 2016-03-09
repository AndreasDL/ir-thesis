from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import pearsonr
from personLoader import dump,load
from sklearn.preprocessing import normalize

from multiprocessing import Pool
POOL_SIZE = 6

class AModel:
    def __init__(self,personLoader):
        self.personLoader = personLoader

    def run(self,person):
        return None;

    def optMetric(self,predictions, truths):
        #==accuracy
        acc = 0
        for pred, truth in zip(predictions, truths):
            acc += (pred == truth)

        return acc / float(len(predictions))
class StdModel(AModel):

    def __init__(self,personLoader, max_k):
        AModel.__init__(self,personLoader)
        self.max_k = max_k

    def run(self,person):
        #load data
        X_train, y_train, X_test, y_test = self.personLoader.load(person)

        #init academic loop to optimize k param
        k = 1
        anova_filter = SelectKBest(f_regression)
        lda          = LinearDiscriminantAnalysis()
        anova_lda    = Pipeline([
            ('anova', anova_filter),
            ('lda', lda)
        ])
        anova_lda.set_params(anova__k=k)

        K_CV = KFold(n=len(X_train),
            n_folds=len(X_train),
            random_state=17, #fixed randomseed ensure that the sets are always the same
            shuffle=False
        ) #leave out one validation

        predictions, truths = [], []
        for train_index, CV_index in K_CV: #train index here is a part of the train set
            #train
            anova_lda.fit(X_train[train_index], y_train[train_index])

            #predict
            pred = anova_lda.predict(X_train[CV_index])

            #save for metric calculations
            predictions.extend(pred)
            truths.extend(y_train[CV_index])

        #optimization metric:
        best_acc = self.optMetric(predictions,truths)
        best_k   = k

        #now try different k values
        for k in range(2,self.max_k):
            anova_filter = SelectKBest(f_regression)
            lda          = LinearDiscriminantAnalysis()
            anova_lda    = Pipeline([
                ('anova', anova_filter),
                ('lda', lda)
            ])
            #set k param
            anova_lda.set_params(anova__k=k)

            #leave one out validation to determine how good the k value performs
            K_CV = KFold(n=len(X_train),
                n_folds=len(X_train),
                random_state=17, #fixed randomseed ensure that the sets are always the same
                shuffle=False
            )

            predictions, truths = [], []
            for train_index, CV_index in K_CV: #train index here is a part of the train set
                #train
                anova_lda.fit(X_train[train_index], y_train[train_index])

                #predict
                pred = anova_lda.predict(X_train[CV_index])

                #save for metric calculations
                predictions.extend(pred)
                truths.extend(y_train[CV_index])

            #optimization metric:
            curr_acc = self.optMetric(predictions, truths)
            if curr_acc > best_acc:
                best_acc = curr_acc
                best_k   = k

        #now the k param is optimized and stored in best_k

        #create classifier and train it on all train data
        anova_filter = SelectKBest(f_regression)
        lda          = LinearDiscriminantAnalysis()
        anova_lda    = Pipeline([
            ('anova', anova_filter),
            ('lda', lda)
        ])
        #set k param
        anova_lda.set_params(anova__k=best_k)
        anova_lda.fit(X_train, y_train)

        predictions = anova_lda.predict(X_test)

        #self.reporter.genReport(person,predictions,y_test,anova_lda)

        return {
            'predictions' : predictions,
            'truths'      : y_test,
            'feat_list'   : anova_lda.named_steps['anova'].get_support(),
            'feat_names'  : self.personLoader.featureExtractor.getFeatureNames()
        }

#correlation not good enough
class CorrelationsAnalyticsModel(AModel):
    def __init__(self, personLoader):
        AModel.__init__(self,personLoader)

    def optMetric(self,predictions, truths):
        return None

    def run(self,person):

        #load all features & keep them in memory
        X_train, y_train = self.personLoader.load(person)

        #each feature separately
        featNames = self.personLoader.featureExtractor.getFeatureNames()
        featCorrelations = []

        s = 'person: ' + str(person) + ' - '
        for index, feat in enumerate(featNames):
            corr = pearsonr(X_train[:, index], y_train)
            featCorrelations.append(corr)

            s += feat + ': ' + str(corr) + ' | '

        print(s)

        return {
            'feat_corr'         : featCorrelations,
            'feat_names'        : featNames,
            #'feat_values'       : X_train,
            'labels'            : y_train,
            'classificatorName' : self.personLoader.classificator.name,
        }
class CorrelationsClusteringsModel(AModel):
    def __init__(self, personLoader):
        AModel.__init__(self,personLoader)

    def optMetric(self,predictions, truths):
        return None

    def run(self):
        feat_corr = []
        for person in range(1,33):
            print('getting correlations for person' + str(person))
            #load all features & keep them in memory
            X_train, y_train = self.personLoader.load(person)

            featNames = self.personLoader.featureExtractor.getFeatureNames()

            persCorr = []
            for index, feat in enumerate(featNames):
                corr = pearsonr(X_train[:, index], y_train)
                persCorr.append(corr[1])

            feat_corr.append(persCorr)

        return feat_corr
class CorrelationsSelectionModel(AModel):
    def __init__(self, cont_personLoader):
        AModel.__init__(self,cont_personLoader)
        self.max_k = 7

    def run(self,person):
        print('starting on person ' + str(person))

        #load all features & keep them in memory
        X_cont, y_cont = self.personLoader.load(person)
        y_lbl = np.array( y_cont )
        y_lbl[ y_lbl <= 5 ] = 0
        y_lbl[ y_lbl >  5 ] = 1

        featNames = self.personLoader.featureExtractor.getFeatureNames()

        #split train / test
        #n_iter = 1 => abuse the shuffle split, to obtain a static break, instead of crossvalidation
        sss = StratifiedShuffleSplit(y_lbl, n_iter=1, test_size=0.25, random_state=19)
        for train_set_index, test_set_index in sss:
            #labels
            X_train, y_train = X_cont[train_set_index], y_lbl[train_set_index]
            X_test , y_test  = X_cont[test_set_index] , y_lbl[test_set_index]

            #correlations are based on the continuous values
            y_train_cont = y_cont[train_set_index]
            y_test_cont  = y_cont[train_set_index]


        #get correlations
        featCorrelations = [] #list[person] = {feat_index => , feat_corr => , feat_name => }
        for index, feat in enumerate(featNames):
            corr = pearsonr(X_train[:, index], y_train_cont)

            featCorrelations.append( {
                'feat_index' : index,
                'feat_corr'  : corr[0],
                'feat_name'  : featNames[index]
            })

        #sort correlations
        featCorrelations.sort(key=lambda tup: tup['feat_corr'], reverse = True) #sort on correlation (in place)
        #sort X_train in same order
        X_train_sorted = []
        for index,video in enumerate(X_train):
            X_train_sorted.append([])
            for map in featCorrelations:
                X_train_sorted[index].append(video[map['feat_index']])
        X_train_sorted = np.array(X_train_sorted)

        X_test_sorted = []
        for index,video in enumerate(X_test):
            X_test_sorted.append([])
            for map in featCorrelations:
                X_test_sorted[index].append(video[map['feat_index']])
        X_test_sorted = np.array(X_test_sorted)


        #academic loop
        featAccuracies = []

        #get lda accuracy for 2 features
        k = 2
        lda = LinearDiscriminantAnalysis()


        #leave out one validation
        K_CV = KFold(n=len(X_train_sorted),
            n_folds=len(X_train_sorted),
            random_state=17, #fixed randomseed ensure that the sets are always the same
            shuffle=False
        )
        predictions, truths = [], []
        for train_index, CV_index in K_CV: #train index here is a part of the train set
            #train
            lda.fit(X_train_sorted[train_index, 0:k], y_train[train_index])

            #predict
            pred = lda.predict(X_train_sorted[CV_index, 0:k])

            #save for metric calculations
            predictions.extend(pred)
            truths.extend(y_train[CV_index])

        best_acc = self.optMetric(predictions,truths)
        best_k   = k
        featAccuracies.append(best_acc)
        print('[' + str(person) + '] k= ' + str(k) + ' acc= ' + str(round(best_acc,3)))



        #try to improve the results with additional metrics
        k += 1

        while ( k <= self.max_k ):
            lda = LinearDiscriminantAnalysis()

            #leave out one validation
            K_CV = KFold(n=len(X_train_sorted),
                n_folds=len(X_train_sorted),
                random_state=17, #fixed randomseed ensure that the sets are always the same
                shuffle=False
            )
            predictions, truths = [], []
            for train_index, CV_index in K_CV: #train index here is a part of the train set
                #train
                lda.fit(X_train_sorted[train_index, 0:k], y_train[train_index])

                #predict
                pred = lda.predict(X_train_sorted[CV_index, 0:k])

                #save for metric calculations
                predictions.extend(pred)
                truths.extend(y_train[CV_index])

            curr_acc = self.optMetric(predictions,truths)
            featAccuracies.append(curr_acc)

            print('[' + str(person) + '] k= ' + str(k) + ' acc= ' + str(round(curr_acc,3)))

            if curr_acc > best_acc :
                best_acc = curr_acc
                best_k   = k

            k += 1

        #amount of features is now optimized, its results is stored in best_acc its value is stored in best_k
        #the acc leading up to it are stored in featAccuracies

        #train the optimized model on all data
        lda = LinearDiscriminantAnalysis()
        #train
        lda.fit(X_train_sorted[:, 0:best_k], y_train)
        #predict
        pred = lda.predict(X_test_sorted[:, 0:best_k])

        #get test accuracy
        test_acc = self.optMetric(pred,y_test)


        return {
            'feat_corr'         : featCorrelations,
            'feat_acc'          : featAccuracies,
            'test_acc'          : test_acc,
            'train_acc'         : best_acc,
            'best_k'            : best_k,
            'feat_names'        : featNames,
            'max_k'             : self.max_k,
            'classificatorName'  : self.personLoader.classificator.name
        }

#random forest more reliable
class GlobalRFAnalyticsModel(AModel):
    def __init__(self, personCombiner,personList):
        AModel.__init__(self,personCombiner)
        self.personList = personList


    def run(self,criterion):
        #http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        classificatorName = str(self.personLoader.classificator.name)

        #load all features & keep them in memory
        y = load('global_y_allpersons' + classificatorName)
        if y == None:
            print('[Warn] Rebuilding cache')
            X, y = self.personLoader.load(personList=self.personList)
            dump(X,'global_X_allpersons')
            dump(y,'global_y_allpersons' + classificatorName)
        else:
            X = load('global_X_allpersons')

        normalize(X,copy=False)

        #grow forest
        forest = RandomForestClassifier(
            n_estimators=5000,
            max_features='auto',
            criterion=criterion,
            n_jobs=-1,
            random_state=0
        )

        #fit forest
        forest.fit(X,y)

        #get importances
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)

        indices = np.argsort(importances)[::-1]
        featNames = self.personLoader.featureExtractor.getFeatureNames()

        return {
                'classificatorName'  : classificatorName,
                'featNames'          : featNames,
                'global_importances' : importances,
                'global_std'         : std,
                'global_indices'     : indices, #[ index of first, index of second] ...
                'criterion'          : criterion
                }
class GlobalRFSelectionModel(AModel):
    def __init__(self, personCombiner):
        AModel.__init__(self,personCombiner)

    def getIntermediateResult(self,criterion,X,y, descr):
        predictions, truths = [], []
        forest = RandomForestClassifier(
            n_estimators=5000,
            max_features='auto',
            criterion=criterion,
            n_jobs=-1,
            random_state=0
        )
        for train_index, CV_index in KFold(n=len(X), n_folds=10, random_state=17, shuffle=False): #train index here is a part of the train set
            #train
            forest.fit(X[train_index], y[train_index])

            #predict
            pred = forest.predict(X[CV_index])

            #save for metric calculations
            predictions.extend(pred)
            truths.extend(y[CV_index])

            result = self.optMetric(predictions=predictions, truths=truths)

        print(descr, ' ', result)

        return result

    def step1(self,criterion,X,y):
        #step 1 rank features
        print('Feature Ranking')
        forest = RandomForestClassifier(
            n_estimators=5000,
            max_features='auto',
            criterion=criterion,
            n_jobs=-1,
            random_state=0
        )
        forest.fit(X,y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
        indices = np.array(np.argsort(importances)[::-1])

        return importances, std, indices
    def step2(self,indices):
        #step 2 eliminate lowest 20%
        print("elimination")
        new_indices = np.array(indices[:int(0.8*len(indices))])

        return new_indices
    def step3(self, criterion, indices, X, y, max_count, feat_names):
        print("building")
        acc_list      = []

        forest = RandomForestClassifier(
            n_estimators=5000,
            max_features='auto',
            criterion=criterion,
            n_jobs=-1,
            random_state=0,
            bootstrap=True,
            oob_score=True
        )

        forest.fit(X[:,indices[0:2]],y)
        best_metric = forest.oob_score_
        acc_list.append(best_metric)
        best_count  = 2
        print("acc with 2 features " , best_metric)

        curr_count = 3
        while curr_count < max_count: #for curr_count in range(2,max_feat):
            forest.fit(X[:,indices[0:curr_count]],y)

            curr_metric = forest.oob_score_
            acc_list.append(curr_metric)

            print("acc with ", curr_count, " : " , curr_metric)

            if curr_metric <= best_metric:
                best_metric = curr_metric
                best_count  = curr_count
            #else: break

            curr_count += 1

        return acc_list, best_count, best_metric

    def run(self,criterion):
        #http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        classificatorName = str(self.personLoader.classificator.name)
        featNames = np.array(self.personLoader.featureExtractor.getFeatureNames())

        #load all features & keep them in memory
        y = load('global_y_allpersons' + classificatorName)
        if y == None:
            print('[Warn] Rebuilding cache')
            X, y = self.personLoader.load()
            dump(X,'global_X_allpersons')
            dump(y,'global_y_allpersons' + classificatorName)
        else:
            X = load('global_X_allpersons')

        self.getIntermediateResult(criterion,X,y,'all features')

        #step1
        importances, std, indices = self.step1(criterion,X,y)
        #step2
        indices = self.step2(indices)
        self.getIntermediateResult(criterion,X[:,indices],y,'80%')

        #step 3 add features one by one
        acc_list, best_count, best_metric = self.step3(criterion,indices,X,y,10,featNames)



        return {
                'classificatorName'  : classificatorName,
                'featNames'          : self.personLoader.featureExtractor.getFeatureNames(),
                'global_importances' : importances,
                'global_std'         : std,
                'global_indices'     : indices, #[ index of first, index of second] ...
                'criterion'          : criterion
                }
class RFClusterModel(AModel):
    def __init__(self, personLoader):
        AModel.__init__(self,personLoader)

    def run(self,person, criterion):
        #http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        classificatorName = str(self.personLoader.classificator.name)

        #load all features & keep them in memory
        y = load('global_y_per_person' + classificatorName +'_p' + str(person))
        if y == None:
            print('[Warn] Rebuilding cache')
            X, y = self.personLoader.load(person)
            dump(X,'global_X_per_person_p' + str(person))
            dump(y,'global_y_per_person' + classificatorName + '_p' + str(person))
        else:
            X = load('global_X_per_person_p' +str(person))

        #grow forest
        forest = RandomForestClassifier(
            n_estimators=5000,
            max_features='auto',
            criterion=criterion,
            n_jobs=-1,
            random_state=0
        )

        normalize(X,copy=False)

        #fit forest
        forest.fit(X,y)

        #get importances
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)

        indices = np.argsort(importances)[::-1]
        featNames = self.personLoader.featureExtractor.getFeatureNames()

        return {
                'classificatorName'  : classificatorName,
                'featNames'          : featNames,
                'importances'        : importances,
                'std'                : std,
                'indices'            : indices, #[ index of first, index of second] ...
                'criterion'          : criterion
                }

class RFSinglePersonModel(AModel):
    def __init__(self, personLoader,person,criterion, treeCount):
        AModel.__init__(self,personLoader)

        classificatorName = str(self.personLoader.classificator.name)

        #load all features & keep them in memory
        self.y = load('global_y_per_person' + classificatorName +'_p' + str(person))
        if self.y == None:
            print('[Warn] Rebuilding cache')
            self.X, self.y = self.personLoader.load(person)
            dump(self.X,'global_X_per_person_p' + str(person))
            dump(self.y,'global_y_per_person' + classificatorName + '_p' + str(person))
        else:
            self.X = load('global_X_per_person_p' +str(person))

        normalize(self.X,copy=False)

        self.treeCount = treeCount
        self.criterion = criterion

    def getImportances(self):
        #grow forest
        forest = RandomForestClassifier(
            n_estimators=5000,
            max_features='auto',
            criterion=self.criterion,
            n_jobs=-1,
            random_state=0
        )

        #fit forest
        forest.fit(self.X,self.y)

        #get importances
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)

        zipper = lambda feat,imp,std: [list(a) for a in zip(feat,imp,std)]

        #returns list of <featExtr, importance, std>
        return zipper(self.personLoader.featureExtractor.featureExtrs,importances,std)
    def filterFeatures(self,to_keep):
        self.X = self.X[:,to_keep]
    def getOOBErrors(self,count):
         #grow forest
        forest = RandomForestClassifier(
            n_estimators=2000,
            max_features='auto',
            criterion=self.criterion,
            n_jobs=-1,
            random_state=0,
            oob_score=True,
            bootstrap=True
        )


        #fit forest
        forest.fit(self.X[:count],self.y)
        oob_score = np.average( forest.oob_score_ )

        return oob_score

    def sortFeatures(self,indices):
        self.X = self.X[indices]

class RFModel(AModel):
    def __init__(self, personLoader, criterion, treeCount,threshold):
        AModel.__init__(self,personLoader)
        self.criterion = criterion
        self.treeCount = treeCount
        self.threshold = threshold

        self.classifiers = []
        self.used_blocks = []

    def getImportance(self,person):
        #importances = np.array([ c.getImportances() for c in self.classifiers])
        return self.classifiers[person-1].getImportances()
    def getOOBErrors(self,person):
         #oob_errors = np.array([ c.getOOBErrors(self.block_count,self.used_blocks) for c in self.classifiers ]) #give errors for current block and previously selected blocks
        return self.classifiers[person-1].getOOBErrors(self.count)

    def run(self):

        #create 32 classifiers
        print('initialising the 32 classifiers ...')
        stop_person = 33
        for person in range(1,stop_person):
            self.classifiers.append(
                RFSinglePersonModel(self.personLoader,person,self.criterion,self.treeCount)
            )

        #step 1: variable ranking: get all importances of the different features
        print('getting importances')
        pool = Pool(processes=POOL_SIZE)
        importances = np.array( pool.map( self.getImportance, range(1,stop_person) ))
        pool.close()
        pool.join()
        avg_importances = np.average(importances[:,:,1],axis=0) #average of each importance

        #step 2: elimintaion: remove everything below threshold, a.k.a. keep everything above the threshold
        print('filtering features -  threshold')
        to_keep = [i for i, val in enumerate(avg_importances) if val > self.threshold]
        for c in self.classifiers: c.filterFeatures(to_keep)
        avg_importances = avg_importances[to_keep]

        #sort features
        indices = np.array(np.argsort(avg_importances)[::-1])
        for c in self.classifiers: c.filterFeatures(indices)

        #add features one by one and select smallest, lowest oob model
        print('building tree')
        lowest_error = 1
        lowest_index = 0
        for i in range(len(indices)):
            self.count = i

            pool = Pool(processes=POOL_SIZE)
            oob_errors = pool.map( self.getOOBErrors, range(1,stop_person) )
            pool.close()
            pool.join()
            avg_oob = np.average(oob_errors, axis=0)

            print("feat Count: " + str(i) + " - error: " + str(avg_oob))

            if avg_oob < lowest_error:
                #TODO std
                lowest_error = avg_oob
                lowest_index = i

        #select smalest tree with lowest error => lowest index & lowest error
        for c in self.classifiers: c.filterFeatures( [c for c in range(lowest_index)] )

        print('final building phase')
        #restart with an empty tree and add features one by one, only keep features that decrease the error with a certain threshold
        prev_oob = 1
        used_features = []
        for i in range(len(lowest_index)):
            self.count = i

            pool = Pool(processes=POOL_SIZE)
            oob_errors = pool.map( self.getOOBErrors, range(1,stop_person) )
            pool.close()
            pool.join()

            avg_oob = np.average(oob_errors, axis=0)
            print("feat Count: " + str(i) + " - error: " + str(avg_oob))

            if avg_oob + self.threshold < prev_oob:
                prev_oob = avg_oob
                used_features.append(i)
