# detecting-metastasis-in-cancer-ml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
url='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
df=pd.read_csv(url,header=None)
x=df.iloc[:,2:]
y=df.iloc[:,1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=1)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
rfc=RandomForestClassifier(n_estimators=100,random_state=1)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print('Accuracy:',accuracy)
df['mean_dimension']=df.iloc[:,2:12].mean(axis=1)
df_sorted=df.sort_values(by='mean_dimension',ascending=False)
print('Sorted tumors based on the likelihood of metastasis:',df_sorted.iloc[:,:2])


rfc=RandomForestClassifier(n_estimators=100,random_state=1)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print('Accuracy:',accuracy)
df['mean_dimension']=df.iloc[:,2:12].mean(axis=1)
df_sorted=df.sort_values(by='mean_dimension',ascending=False)
print('Sorted tumors based on the likelihood of metastasis:',df_sorted.iloc[:,:2])
