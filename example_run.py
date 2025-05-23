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
