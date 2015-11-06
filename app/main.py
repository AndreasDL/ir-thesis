import dataLoader as DL
import models
import plotters

def main_single():
	test_size = 4
	used_person = 7

	#load dataset
	(X_train, y_train, X_test, y_test) = DL.loadSinglePersonData(person=used_person, test_size=test_size)

	#classify
	train_acc, test_acc, clf = models.linSVM(X_train,y_train, X_test,y_test)

	#scores
	print('model: lin SVM',
		'\n\tTrain accuracy: ', train_acc,
		'\n\tTest accuracy: ' , test_acc
	)

def main_All():
	test_size = 4
	for person in range(1,32):
		#load dataset
		(X_train, y_train, X_test, y_test) = DL.loadSinglePersonData(person=person, test_size=test_size)

		#classify
		train_acc, test_acc, clf = models.linSVM(X_train,y_train, X_test,y_test)

		#scores
		print('Person: ', person,
			'\tmodel: lin SVM',
			'\tTrain accuracy: ', train_acc,
			'\tTest accuracy: ' , test_acc
		)

if __name__ == "__main__":
	main_All()