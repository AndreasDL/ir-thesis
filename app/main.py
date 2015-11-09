import dataLoader as DL
import featureExtractor as FE
import models
import plotters
import numpy as np



left_channels_to_try  = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3']
right_channels_to_try = ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4']

def main_single():
	test_size = 8
	used_person = 1

	#load dataset
	print('loading')
	#quick load 
	#(X_train, y_train, X_test, y_test) = DL.load('person7TS4LMRClass')
	
	#slow load + dump
	(X_train, y_train, X_test, y_test) = DL.loadSinglePersonData(person=used_person, test_size=test_size)
	#DL.dump(X_train, y_train, X_test, y_test, 'person7TS4LMRClassSplit')

	#classify
	train_acc, test_acc, clf = models.linSVM(X_train,y_train, X_test,y_test)

	#scores
	print('model: lin SVM',
		'\n\tTrain accuracy: ', train_acc,
		'\n\tTest accuracy: ' , test_acc
	)
def main_all_one_by_one():
	test_size = 0.25
	avg_train, avg_test = 0, 0

	for left, right in zip(left_channels_to_try, right_channels_to_try):
		print('left: ', left, ' - right: ', right)

		#generate the extraction function
		def func(samples):
			return FE.LMinRFraction(samples, left_channel=left, right_channel=right)

		for person in range(1,33):
			#load dataset
			(X_train, y_train, X_test, y_test) = DL.loadSinglePersonData(featureFunc=func, person=person, test_size=test_size)

			#classify
			train_acc, test_acc, clf = models.linSVM(X_train,y_train, X_test,y_test)
			avg_test  += test_acc
			avg_train += train_acc

			#scores
			#print('Person: ', person,
			#	'\tmodel: lin SVM',
			#	'\tTrain accuracy: ', train_acc,
			#	'\tTest accuracy: ' , test_acc
			#)

		avg_test  /= 32
		avg_train /= 32

		print('avg\t',
			'\tTrain accuracy: ', train_acc,
			'\tTest accuracy: ' , test_acc
		)

def main_all():
	test_size = 4

	#load dataset
	print('loading')
	#quick load 
	#(X_train, y_train, X_test, y_test) = DL.load('allpersonsTS4LMRClass')
	
	#slow load + dump
	(X_train, y_train, X_test, y_test) = DL.loadMultiplePersonsData(test_size=test_size)
	#DL.dump(X_train, y_train, X_test, y_test, 'allpersonsTS4LMRClass')

	#classify
	print('training')
	train_acc, test_acc, clf = models.linSVM(X_train,y_train, X_test,y_test)

	#scores
	print('model: lin SVM',
		'\n\tTrain accuracy: ', train_acc,
		'\n\tTest accuracy: ' , test_acc
	)


if __name__ == "__main__":
	main_all_one_by_one()