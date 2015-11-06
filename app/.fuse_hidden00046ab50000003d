import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_validation import KFold

def linReg(X_train, y_train, X_test, y_test):
    #perform linear regression
    
    #Create linear regression object that uses all cores
    regr = LinearRegression(n_jobs=-1)

    # Train the model using the training sets
    regr.fit(X_train, y_train)
    
    #MSE train err
    train_err = np.mean((regr.predict(X_train) - y_train) ** 2)

    #MSE for test set
    test_err = -1
    if len(X_test) != 0 and len(y_test) != 0:
        test_err  = np.mean( (regr.predict(X_test) - y_test)**2 )

    return train_err, test_err, regr

def ridgeReg(X_train, y_train, X_test, y_test, cvSets= 8):
    #linear regression with I2 regularisation

    if cvSets <= 1:
        print("a minimum of 2 cv sets is required to actually have a train and validation set")
        exit(-1)
        
    #perform linear regression
    alphaValues = [0.01,0.03,0.1,0.3,1,3,10,20,30,50,70,100,200,300]

    #get sets
    K_CV = KFold(len(X_train),n_folds=cvSets,random_state=17, shuffle=True)
    best_err = float('inf')
    best_regr = None
    for i in range(len(alphaValues)):
        err = 0
        for train_index, CV_index in K_CV:
            regr = Ridge( alpha=alphaValues[i] ) #Create linear regression object
            
            regr.fit( X_train[train_index], y_train[train_index] )
            err += np.mean( (regr.predict(X_train[CV_index]) - y_train[CV_index])**2 )

        if err < best_err:
            best_err = err
            best_regr = regr
            best_alpha = alphaValues[i]
    
    train_err = best_err
    
    test_err = -1
    if len(X_test) != 0 and len(y_test) != 0:
        test_err  = np.mean( (regr.predict(X_test) - y_test)**2 )

    #print('ridgeReg lowest alpha: ', best_alpha)

    return train_err, test_err, regr