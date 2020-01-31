# -*- coding: utf-8 -*-

#--------------------------------------------
# Importing the Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#--------------------------------------------
# Getting the Data

data = pd.read_csv('KNN_Project_Data')

# Let's get a feel for the data
# It seems like we have 11 columns and with 1000 values each
data.head()
data.info()
data.describe()

#--------------------------------------------
# Data Exploration

# Since this data is artificial, we can just create a pairplot
# This is a large plot with 11 columns and 1000 points each so it will take some time
sns.pairplot(data=data, hue='TARGET CLASS', palette='coolwarm')

#--------------------------------------------
# Standardizing the Data

# It is best practice to standardize data before plugging it into a KNN classifier
scaler = StandardScaler()

scaler.fit(data.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(data.drop('TARGET CLASS', axis=1))

# This is mainly to check whether everything scaled properly
data_feat = pd.DataFrame(scaled_features, columns=data.columns[:-1])
data_feat

#--------------------------------------------
# Train Test Split

# With our values standardized, let's assign them to our variables for splitting
X = scaled_features
y = data['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#--------------------------------------------
# K Nearest Neighbors

# Now let's train and test the KNN classifier and see what we get
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

# It seems like the KNN classifier is not performing optimally at 70% accuracy
# Let's see if we can improve by selecting a better K value
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

#--------------------------------------------
# Choosing K Values

# These lines test for the K value with the lowest error rate from numbers 1-40
# We will plot this out and hand pick the best values according to the graphy
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# According to the graph it seems like 7 may be a good value to choose with a low error rate
plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', 
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

#--------------------------------------------
# Retraining with the new K Value

# Let's retrain the model with a K value of 7
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# It seems the like the model improved with an improved accuracy of 82%
# For this being a KNN it seems like these results are sufficient enough to leave it here
print('With K = 7')
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))





# Project Source: Pierian Data
