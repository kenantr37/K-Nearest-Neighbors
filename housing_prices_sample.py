# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:32:14 2021

@author: Zeno
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # to split the data as test and train
from sklearn.preprocessing import StandardScaler # to normalize the data
from sklearn.neighbors import KNeighborsClassifier # to create KNN model
# I wanted to predict median_house_value by using other features with KNN algorithm
data = pd.read_csv("D:/Machine Learning Works/K-Neirest Neighbour (KNN)/housing.csv")
data.dropna(inplace = True) # There are a few NaN values our features
data.drop(["ocean_proximity"],axis = 1,inplace = True) # I drop the last column 'cause I won't use it
# y is median_house_value
y = data.median_house_value
# x is other features
x = data.drop(["median_house_value"],axis = 1 ,inplace = False)
# Now, we can normalize x features' values'
x_normalized = StandardScaler().fit(x).transform(x)
# splitting data as train and test
x_test,x_train,y_test,y_train = train_test_split(x,y,test_size = 0.2,random_state = 1 )
# Let's create KNN model
# I decided 3 neighbours for initial value but I'll look which value is much more proper for neigbour value
knn_model = KNeighborsClassifier(n_neighbors = 3).fit(x_train,y_train)
# for the last thing, we can look at the score of the accuracy
print("for neighbour value is {}, score is {} ".format(3,knn_model.score(x_test,y_test)))
# We need to find the best n value and for this we could make a loop
accuracy_score=[] #to see which n value is better
for each in range(500,509):
    knn_model_2 = KNeighborsClassifier(n_neighbors=each).fit(x_train,y_train)
    prediction_2 = knn_model_2.predict(x_test)
    print("when n is {}, score is {} ".format(each,knn_model_2.score(x_test,y_test)))
    accuracy_score.append(knn_model_2.score(x_test,y_test))
plt.plot(range(1,10),accuracy_score)
plt.xlabel("range")
plt.ylabel("accuracy")
plt.show()