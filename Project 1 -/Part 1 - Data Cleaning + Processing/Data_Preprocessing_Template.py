# -*- coding: utf-8 -*-
# Data Processing Template

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('data.csv')
# Setting Independent and Dependent Variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Missing Data
from sklearn.preprocessing import Imputer
# Replace missing data with set method
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Select column to fit missing data 
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Select column to encode 
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
# Assign encoder to select independent variable
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
# Affects dependent variable from assignment
y = labelencoder_X.fit_transform(y)

