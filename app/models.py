import personLoader
import featureExtractor
import classificators
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class AModel:
    def __init__(self,personLoader):
        self.personLoader = personLoader

    def run(self):
        return None;

    def optMetric(self,predictions, truths):
        return 0

class StdModel(AModel):

    def __init__(self,personLoader, max_k):
        AModel.__init__(self,personLoader)
        self.max_k = max_k

    def optMetric(self,predictions, truths):
        #==accuracy
        acc = 0
        for pred, truth in zip(predictions, truths):
            acc += (pred == truth)

        return acc / float(len(predictions))

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
            x_temp = X_train[train_index]
            y_temp = y_train[train_index]
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