# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:32:27 2020

@author: Gurkan
"""

import keras
from keras.layers import Dense
from keras.models import Sequential

import pandas as pd
from sklearn import preprocessing

veriler=pd.read_csv("veri.csv")

x=veriler.iloc[:,3:13].values
y=veriler.iloc[:,13:].values

le=preprocessing.LabelEncoder()
x[:,1]=le.fit_transform(x[:,1])

le2=preprocessing.LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe=ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])], 
                      remainder="passthrough")

x=ohe.fit_transform(x)
x=x[:,1:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


classifier=Sequential()
classifier.add(Dense(6, activation="relu",input_dim=11))
classifier.add(Dense(6, activation="relu"))
classifier.add(Dense(1, activation="sigmoid"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(x_train,y_train,epochs=150)

y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
print(cm)












