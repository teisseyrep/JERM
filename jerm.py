from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import numpy as np
from utils import sigma
from cvxopt import matrix
from cvxopt.solvers import qp

def grad_logLike(beta,X,s,ex):
    n = X.shape[0]
    ex_matrix = ex.reshape(n, 1)
    s_matrix = s.reshape(n,1)
    X1 = np.append(np.ones((n,1)),X, axis=1)
    eta = np.dot(X1,beta)
    sigma1 = sigma(eta)
    var1 = sigma1*(1-sigma1)
    a= var1*( (s_matrix-ex_matrix*sigma1)/(sigma1*(1-ex_matrix*sigma1)) )
    X1t = np.transpose(X1)
    res = np.dot(X1t,a)
    return(res)


def logistic_fit_mm(X,s,ex,mm_iter,mm_tol,mm_reg,beta_old):
    n = X.shape[0]
    X1 = np.append(np.ones((n,1)),X, axis=1)
    X1t = np.transpose(X1)
    p = X1.shape[1]
    P = 0.25*np.dot(X1t,X1) + np.diag(np.ones(p))*mm_reg
    Pm = matrix(P)
    #beta_old = np.zeros((p,1))
    for iter in np.arange(0,mm_iter):
        q = -grad_logLike(beta_old,X,s,ex)
        qm = matrix(q)
        solve = qp(Pm,qm)
        beta_new = np.array(solve['x']) + beta_old
        if np.any(np.isnan(beta_new)):
            beta_new[np.where(np.isnan(beta_new))]= 0
        if np.max(np.abs(beta_new-beta_old))<0.01:
            break
        beta_old = beta_new
        
        
        
    return beta_old

class JERM(BaseEstimator):
   
    def __init__(self, clf_init,clf_ex,epochs=100,mm_iter=200,mm_tol=0.001,mm_reg=0.01):
        self.clf_init = clf_init
        self.clf_ex = clf_ex
        self.epochs = epochs
        self.mm_iter = mm_iter
        self.mm_tol = mm_tol
        self.mm_reg = mm_reg
        self.beta= None
        self.logLik = None
        
    def fit(self, X, s):
        
        model_naive = self.clf_init
        model_naive.fit(X,s)
        sx = model_naive.predict_proba(X)[:,1]
        ex = (sx+1)/2
        n = X.shape[0]
        X1 = np.append(np.ones((n,1)),X, axis=1)
        p = X1.shape[1]
        beta_old = np.zeros((p,1))
        logLik = np.zeros(self.epochs)
    
        
  
        w0 =  np.where(s==0)[0]
        w1 =  np.where(s==1)[0]
        Xs0 = X[w0,:]
        Xs1 = X[w1,:]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Xs0)
        distances, indices = nbrs.kneighbors(Xs1)
        sel_up = indices.flatten() #Spies: S=0 observations which are the NN of some observations from S=1.
    
        for i in np.arange(self.epochs):
        # Model for posterior probability:
            beta=  logistic_fit_mm(X,s,ex,mm_iter=self.mm_iter,mm_tol=self.mm_tol,mm_reg=self.mm_reg,beta_old=beta_old)
            eta = np.dot(X1,beta)
            yx = sigma(eta)
            beta_old = beta
            
         # Model for propensity score:  
            yx_matrix = yx.reshape((n,1))
            ex_matrix = ex.reshape((n,1))
            w0 =  np.where(s==0)[0]
            w1 =  np.where(s==1)[0]
            yx0 = yx_matrix[w0]
            ex_matrix0 = ex_matrix[w0]
            yx_cond0 = yx0*(1-ex_matrix0)/(1-yx0*ex_matrix0) #P(Y=1|X=x,S=0)
            thr_opt = np.min(yx_cond0[sel_up])
            sel0 = np.where(yx_cond0>thr_opt)[0] 
            if sel0.shape[0]<10 and w0.shape[0]>10:
                sel0 = np.argsort(-yx0[:,0])[1:10]
            sel1 = w0[sel0]
            sel = np.union1d(sel1,w1)                     
         
       
            
            if sel.shape[0]>0:
                Xsel = X[sel,:]
                ssel = s[sel]
            else:
                Xsel = X
                ssel = s
                   
            if np.mean(ssel)==0:
                ssel[np.random.choice(ssel.shape[0],2,replace=False)]=1    
            if np.mean(ssel)==1:
                ssel[np.random.choice(ssel.shape[0],2,replace=False)]=0    
                
                    
            self.clf_ex.fit(Xsel,ssel)
            ex = self.clf_ex.predict_proba(X)[:,1] 
            
            yx_matrix = yx.reshape((n,1))
            ex_matrix = ex.reshape((n,1))
           
            s_matrix = s.reshape((n,1))
            logLik[i] = np.mean(s_matrix*np.log(yx*ex) + (1-s_matrix)*np.log(1-yx*ex))
            
        self.beta = beta
        self.logLik = logLik    
        
        
        return self

        
    def predict_proba(self, Xtest):
        n = Xtest.shape[0]
        Xtest1 = np.append(np.ones((n,1)),Xtest, axis=1)
        prob = sigma(np.dot(Xtest1,self.beta))
        return prob
    
    def compute_propensity(self,Xtest):
        ex_test = self.clf_ex.predict_proba(Xtest)[:,1]
        return ex_test    
    
          

