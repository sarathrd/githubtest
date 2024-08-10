import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
dataset=pd.read_csv("reg4.csv")
print(dataset)
X=dataset.iloc[:,[1,2,3,4,5,6]].values
print(X)
lab={"Yes":1,"No":0}
dataset["targetlabel"]=dataset.target.map(lab)
print(dataset)
Y=dataset.iloc[:,7].values
print(Y)
sc_X=StandardScaler()
X1=sc_X.fit_transform(X)
print(X1)
lr=LogisticRegression(random_state=0)
lr.fit(X1,Y)
Y_pred=lr.predict(X1)
print(Y_pred)
# for checking purpose only
print(Y)
t=pd.DataFrame(Y_pred,Y)
print(t)
cm=confusion_matrix(Y,Y_pred)
print(cm)
