import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def mi_filter(X,y,pmax=50):
    mi = np.zeros(X.shape[1])
    for j in np.arange(X.shape[1]):    
        mi[j] = mutual_info_classif(X[:,j].reshape(-1,1), y)
    sel = np.argsort(-mi)[0:pmax]
    return sel

def sigma(x):
    res = np.exp(x)/(1+np.exp(x))
    return res

def remove_collinear(X,threshold=0.9): 

    X = pd.DataFrame(X)
    X_corr = X.corr()
    
    columns = np.full((X_corr.shape[0],), True, dtype=bool)
    for i in range(X_corr.shape[0]):
        for j in range(i+1, X_corr.shape[0]):
            if X_corr.iloc[i,j] >= threshold:
                if columns[j]:
                    columns[j] = False
    selected_columns = X.columns[columns]
    return np.array(selected_columns)

def make_binary_class(y):
   if np.unique(y).shape[0]>2: 
       values, counts = np.unique(y, return_counts=True)
       ind = np.argmax(counts)
       major_class = values[ind]
       for i in np.arange(y.shape[0]):
           if y[i]==major_class:
               y[i]=1
           else:
               y[i]=0    
   return y           


