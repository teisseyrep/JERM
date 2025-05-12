# JERM

This is a Github repository containing code for JERM [1].

JERM is a method of learning from PU (Positive-Unlabeled) data. The method is recommended in situations where the labeling mechanism may depend on the feature vector, but it also works effectively in situations where the labeling mechanism is independent of the feature vector (Selected Completely at random assumption).


## Abstract ##

Learning from positive and unlabeled data (PU learning) is actively researched machine learning task. The goal is to train a binary classification model based on a training dataset containing part of positives which are labeled, and unlabeled instances. Unlabeled set includes remaining part of positives and all negative observations. An important element in PU learning is modeling of the labeling mechanism, i.e. labels’ assignment to positive observations. Unlike in many prior works, we consider a realistic setting for which probability of label assignment, i.e. propensity score, is instance-dependent. In our approach we investigate minimizer of an empirical counterpart of a joint risk which depends on both posterior probability of inclusion in a positive class as well as on a propensity score. The non-convex empirical risk is alternately optimized with respect to parameters of both functions. In the theoretical analysis we establish risk consistency of the minimizers using recently derived methods from the theory of empirical processes. Besides, the important development here is a proposed novel implementation of an optimization algorithm, for which sequential approximation of a set of positive observations among unlabeled ones is crucial. This relies on modified technique of ’spies’ as well as on a thresholding rule based on conditional probabilities. Experiments conducted on 20 data sets for various labeling scenarios show that the proposed method works on par or more effectively than state-of-the-art methods based on propensity function estimation.


Example
--------
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import  make_binary_class, mi_filter, remove_collinear
from sklearn.model_selection import train_test_split

from labelling import make_pu_labels
from jerm import JERM

# Parameters:
ds = "Breast-w" #dataset
label_scheme = "S1" #labelling scenario
c = 0.5 #label frequency P(S=1|Y=1)

# Load dataset:
df_name = ds + '.csv'
df = pd.read_csv(df_name, sep=',')
del df['BinClass']
df = df.to_numpy()
p = df.shape[1]-1
Xall = df[:,0:p]
yall = df[:,p]
yall = make_binary_class(yall)
selected_columns = remove_collinear(Xall,0.95)
Xall = Xall[:,selected_columns]
sel = mi_filter(Xall,yall,pmax=30)
Xall = Xall[:,sel]
X, X_test, y, y_test = train_test_split(Xall, yall, test_size=0.25, random_state=42)


# Create PU dataset:
model_oracle = LogisticRegression()
model_oracle.fit(X,y)
prob_true=model_oracle.predict_proba(X)[:,1]
s, ex_true = make_pu_labels(X,y,prob_true=prob_true,label_scheme=label_scheme,c=c)    

# Run JERM method:
clf_ex = LogisticRegression()
clf_init = LogisticRegression()
model = JERM(clf_init,clf_ex,epochs=100) 
model.fit(X,s)
prob_y = model.predict_proba(X_test) 

#Evaluate JERM:
acc = accuracy_score(y_test, np.where(prob_y>0.5,1,0))
print("JERM, Accuracy=", acc)

```
## References
[1] Wojciech Rejchel, Paweł Teisseyre, Jan Mielniczuk, [Joint empirical risk minimization for instance-dependent positive-unlabeled data](https://www.sciencedirect.com/science/article/pii/S0950705124010785), Knowledge-Based Systems, Volume 304, 2024.


## Contact

If you have any questions or issues, please reach out via my email:

> teisseyrep AT ipipan DOT waw DOT pl
