#the origin of these functions is explained in 01-11-course-project-pre-processing-daata.ipynb

import numpy as np
import pandas as pd

data_path = '../data/ecommerce_data.csv'

def get_data():
    df = pd.read_csv(data_path)
    #convert to matrix
    data = df.values

    #splitting X and Y
    X = data[:, :-1]
    Y = data[:, -1]

    #normalizing a couple numerical columns
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    #the categorical column, time of day
    #manually making one-hot encoding
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
    for n in range(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1
    
    return X2, Y

#for the logistic class we're only using binary data
#e.g. where user action is 0 or 1
#so we're dropping all other rows
def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2