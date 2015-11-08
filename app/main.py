import dataLoader as DL
import models
import plotters
import numpy as np

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
	test_size = 8
	avg_train, avg_test = 0, 0

	for person in range(1,33):
		#load dataset
		(X_train, y_train, X_test, y_test) = DL.loadSinglePersonData(person=person, test_size=test_size)

		#classify
		train_acc, test_acc, clf = models.linSVM(X_train,y_train, X_test,y_test)
		avg_test  += test_acc
		avg_train += train_acc

		#scores
		print('Person: ', person,
			'\tmodel: lin SVM',
			'\tTrain accuracy: ', train_acc,
			'\tTest accuracy: ' , test_acc
		)

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

def count_all_one_by_one():
	test_size = 0

	for person in range(1,33):
		#load dataset
		(X_train, y_train, X_test, y_test) = DL.loadSinglePersonData(person=person, test_size=test_size)

		tr_happyCount = np.sum(y_train)
		tr_sadCount = 40 - test_size - tr_happyCount
		te_happyCount = np.sum(y_test)
		te_sadCount = test_size - te_happyCount

		#scores
		print('Person: ', person,
			'\tTr happy %: ', tr_happyCount/float(32-test_size),
			'\tTe happy %: ', te_happyCount/float(test_size)
		)


if __name__ == "__main__":
	count_all_one_by_one()