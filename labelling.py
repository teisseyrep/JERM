import numpy as np
import warnings   
from utils import sigma
from scipy.stats import cauchy

def make_pu_labels(X,y,prob_true=None,label_scheme='S1',c=0.5):
    
    
    if prob_true.shape[0]!=X.shape[0]:
        raise Exception('The length of prob_true does not match the number of instances in X')
    
    prob_true[np.where(prob_true==1)] = 0.999
    prob_true[np.where(prob_true==0)] = 0.001
    
    n = X.shape[0]
    s = np.zeros(n)
    ex_true = np.zeros(n)    
    
    if label_scheme=='S1':
        for i in np.arange(0,n,1):
            ex_true[i] = c
    elif label_scheme=='S2':
        if any(prob_true)==None:
            raise Exception('Argument prob_true should be specified')           
        lin_pred = np.log(prob_true/(1-prob_true))  
        
        a_seq = np.linspace(start=-10, stop=10, num=100)
        score = np.zeros(100)
        k=0
        w1 = np.where(y==1)[0]
        for a in a_seq:
            for i in np.arange(0,n,1):
                ex_true[i] = sigma(lin_pred[i] + a)
            score[k] = np.abs(np.mean(ex_true[w1])-c)
            k=k+1
        a_opt = a_seq[np.argmin(score)]
        for i in np.arange(0,n,1):
            ex_true[i] = sigma(lin_pred[i] + a_opt)
    elif label_scheme=='S3':
        if any(prob_true)==None:
            raise Exception('Argument prob_true should be specified')           
        lin_pred = np.log(prob_true/(1-prob_true))  
        
        a_seq = np.linspace(start=-10, stop=10, num=100)
        score = np.zeros(100)
        k=0
        w1 = np.where(y==1)[0]
        for a in a_seq:
            for i in np.arange(0,n,1):
                ex_true[i] = cauchy.cdf(lin_pred[i] + a)
            score[k] = np.abs(np.mean(ex_true[w1])-c)
            k=k+1
        a_opt = a_seq[np.argmin(score)]
        for i in np.arange(0,n,1):
            ex_true[i] = cauchy.cdf(lin_pred[i] + a_opt)              
    elif label_scheme=='S4':
        if any(prob_true)==None:
            raise Exception('Argument prob_true should be specified')           
        lin_pred = np.log(prob_true/(1-prob_true))  
        
        a_seq = np.linspace(start=-10, stop=10, num=100)
        score = np.zeros(100)
        k=0
        w1 = np.where(y==1)[0]
        for a in a_seq:
            for i in np.arange(0,n,1):
                ex_true[i] = sigma(lin_pred[i] + a)**10
            score[k] = np.abs(np.mean(ex_true[w1])-c)
            k=k+1
        a_opt = a_seq[np.argmin(score)]
        for i in np.arange(0,n,1):
            ex_true[i] = sigma(lin_pred[i] + a_opt)**10 
    else:
        print('Argument label_scheme is not defined')
                    
    for i in np.arange(0,n,1):
        if y[i]==1:
            s[i]=np.random.binomial(1, ex_true[i], size=1)
        
    if np.sum(s)<=1:
        s[np.random.choice(s.shape[0],2,replace=False)]=1
        warnings.warn('Warning: <2 observations with s=1. Two random instances were assigned label s=1.')
        
    return s, ex_true




def label_transform_MNIST(y):
    w1 = (y%2==0)
    w0 = (y%2!=0)
    y[w1]=1
    y[w0]=0
    return y

def label_transform_CIFAR10(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 1, 8, 9]:
            y[i]=1
        else:
            y[i]=0
    return y
   
def label_transform_USPS(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 1, 2, 3, 4]:
            y[i]=1
        else:
            y[i]=0
    return y
 
def label_transform_Fashion(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 2, 3, 4, 6]:
            y[i]=1
        else:
            y[i]=0
    return y   
    
    



