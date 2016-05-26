Physiological feature selection methods for emotion recognition, a comparative approach.
=====================

# What?
ir-thesis performed in 2015-2016, based on the DEAP dataset. The repo contains both scription and the code used during this thesis.
Note that the code consists of a set of small python scripts that compare different feature selection methods. This code is not style-proof, and might be (is) quite messy. This is the result of fast comparisons.

#Folders:
* ML-Coursera/tasks: the tasks corresponding to the machine learning course of andrew NG. This is a good starting point to learn machine learning and can be found at coursera: https://www.coursera.org/learn/machine-learning

* app: contains the scripts.
	* for which you need the following things:
		* the preprocessed version of the DEAP dataset, for which you need a key. This can be found at http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
		* you may need to adjust the ddpad (where the data is retreived / stored) and the rdpad where the results are stored.
		* python 3, with sklearn
		* a lot of patience
	* subfolders:
		* archive: containing old scripts that are not discussed in the final results
		* scriptie: scripts to generate plots for this thesis
		* the python scripts in the app folder are the used once. The Genscript is for the cross-subject feature selection methods and the PersScript is for the person specific emotion recognition. PersTree is the modified random forest for cross subject feature selection.

* scriptie: containing the written results of this work. 