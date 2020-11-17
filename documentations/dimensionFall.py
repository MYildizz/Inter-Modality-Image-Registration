# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:32:53 2020

@author: Gurkan
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("wine.csv")

x=data.iloc[:,0:13].values
y=data.iloc[:,13:].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from sklearn.decomposition import PCA

pca=PCA(n_components=2)

x_train2=pca.fit_transform(x_train)
x_test2=pca.transform(x_test)


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=2)
classifier.fit(x_train2,y_train)

y_pred=classifier.predict(x_test2)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
print(cm)



# LDA boyut ındırgeme   pca dan daha iyi sonuç verdi

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda=LDA(n_components=2)

x_train3=lda.fit_transform(x_train,y_train.ravel())
x_test3=lda.transform(x_test)

classifier2=LogisticRegression()
classifier2.fit(x_train3,y_train)
y_pred2=classifier2.predict(x_test3)

print(confusion_matrix(y_test,y_pred2))











