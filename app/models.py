import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn as SKL
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from pprint import pprint


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def linReg(x_train,y_train,x_test,y_test):
    #perform linear regression
    
    #Create linear regression object
    regr = SKL.linear_model.LinearRegression(n_jobs=-1)

    # Train the model using the training sets
    regr.fit(x_train, y_train)
    
    train_err = np.mean((regr.predict(x_train) - y_train) ** 2)

    test_err = -1
    if x_test != None and y_test != None:
        test_err  = np.mean( (regr.predict(x_test) - y_test)**2 )

    return train_err, test_err, regr

def ridgeReg(x_train,y_train,x_test,y_test, cvSets= 8):
    if cvSets <= 1:
        print("a minimum of 2 cv sets is required to actually have a train and validation set")
        exit()
        
    #perform linear regression
    alphaValues = [0.01,0.03,0.1,0.3,1,3,10,20,30,50,70,100,200,300]
    cvSize = round( len(x_train) / cvSets )

    #get sets
    err   = np.zeros( len(alphaValues) )
    for j in range(cvSets):

        x = np.concatenate( (x_train[:cvSize * j], x_train[cvSize * (j+1):]), 0 )
        y = np.concatenate( (y_train[:cvSize * j], y_train[cvSize * (j+1):]), 0 )

        x_cv = x_train[cvSize * j: cvSize * (j+1)]
        y_cv = y_train[cvSize * j: cvSize * (j+1)]

        #check the different alpha values
        for i in range(len(alphaValues)):
            alpha = alphaValues[i]

            regr = SKL.linear_model.Ridge(alpha=alpha) #Create linear regression object
            regr.fit(x, y) # Train the model using the training sets

            err[i] += np.mean( (regr.predict(x_cv) - y_cv)** 2 )
    
    train_err = err[np.argmin(err)]
    
    test_err = -1
    if x_test != None and y_test != None:
        test_err  = np.mean( (regr.predict(x_test) - y_test)**2 )
    
    best_alpha = alphaValues[np.argmin(err)]
    regr = SKL.linear_model.Ridge(alpha=best_alpha) #Create linear regression object
    regr.fit(x, y)

    #print('ridgeReg lowest alpha: ', alphaValues[np.argmin(err)])

    return train_err, test_err, regr

def polyReg(x_train,y_train,x_test,y_test, cvSets=8):
    degrees = [2,3,4]

    cvSize = round( len(x_train) / cvSets )

    #get sets
    err   = np.zeros( len(degrees) )
    for j in range(cvSets):

        x = np.concatenate( (x_train[:cvSize * j], x_train[cvSize * (j+1):]), 0 )
        y = np.concatenate( (y_train[:cvSize * j], y_train[cvSize * (j+1):]), 0 )

        x_cv = x_train[cvSize * j: cvSize * (j+1)]
        y_cv = y_train[cvSize * j: cvSize * (j+1)]

        #check the different alpha values
        for i in range(len(degrees)):
            regr = make_pipeline(PolynomialFeatures(degrees[i]), Ridge())
            regr.fit(x, y)

            err[i] += np.mean( (regr.predict(x_cv) - y_cv)** 2 )
    
    best_deg  = degrees[np.argmin(err)]
    train_err = err[np.argmin(err)]

    test_err = -1
    if x_test != None and y_test != None:
        test_err  = np.mean( (regr.predict(x_test) - y_test)**2 )
    
    regr = make_pipeline(PolynomialFeatures(best_deg), Ridge())
    regr.fit(x, y)

    #print('ridgeReg lowest alpha: ', alphaValues[np.argmin(err)])

    return train_err, test_err, regr