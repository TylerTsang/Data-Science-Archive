# -*- coding: utf-8 -*-

#--------------------------------------------------
# Overview

# Data used is a fake advertising data set that indicates whether an internet user clicked on the ad
# The goal is to create a model that predicts whether or not a user clicked on the ad

# Daily Time Spent on Site: Time spent on site in minutes
# Age: Age of the user in years
# Area Income: Average income for the geographical area of the user
# Daily Internet Usage: Average minutes user spends on internet
# Ad Topic Line: Headline of the advertisement
# City: City of the user
# Male: Whether or not the user is a male
# Country: Country of the user
# Timestamp: Time the user clicked on the ad
# Clicked on Ad: Whether the user clicked on the Ad

#--------------------------------------------------
# Importing Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report 

#--------------------------------------------------
# Loading the Data + Quick Overview

# Adding the data
ad = pd.read_csv('advertising.csv')

# Quick overview to get a feel for the data
ad.head()
ad.describe()
ad.info()

#--------------------------------------------------
# Data Exploration + Visualization

# Quickly setting seaborn style
sns.set_style('whitegrid')

# Histogram for age of users
ad['Age'].hist(bins=30)
plt.xlabel('Age')

# Jointplot on age + area income
sns.jointplot(data=ad, x='Age', y='Area Income')

# Jointplot on age and daily time spent on site
# Style is KDE
sns.jointplot(data=ad, x='Age', y='Daily Time Spent on Site', kind='kde')

# Jointplot on daily time spent on site and daily internet usage
sns.jointplot(data=ad, x='Daily Time Spent on Site', y='Daily Internet Usage')

# Now let's do a pairplot for all possible relationships
# Our dataset is relatively small, containing 1000 points a pairplot is feasable
# Let's set the hue to be whether users clicked on the ad
# That way it'll show any relationships for our goal variable
# Interestingly, the pairplot shows daily internet usage may have a correlation to whether users clicked on the ad
sns.pairplot(data=ad, hue='Clicked on Ad')

#--------------------------------------------------
# Train Test Split

list(ad.columns.values)

# Setting X to all numerical values in our dataset
X = ad[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male' ]]
# Setting y to our dependent value, whether users clicked on the ad
y = ad['Clicked on Ad']

# Splitting the data into training and testing sets for our model to learn from
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#--------------------------------------------------
#Logistic Regression

# Training the model on our training set
log = LogisticRegression()
log.fit(X_train, y_train)

# Using out model to predict for the X of our test set
predictions = log.predict(X_test)

#--------------------------------------------------
# Evaluating the Model

# Seems like out model performed well with a 90% accuracy!
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))





# Project Source: Pierian Data