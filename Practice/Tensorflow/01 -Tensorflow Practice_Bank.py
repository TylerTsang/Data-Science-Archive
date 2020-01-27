# -*- coding: utf-8 -*-
#---------------------------------------------------------------
# Importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

#---------------------------------------------------------------
# Importing data file
df = pd.read_csv('bank_note_data.csv')

#---------------------------------------------------------------
# Data Overview

# Data contains 5 columns:
# 1. variance of wavelet transformed image
# 2. skewness of wavelet transformed image
# 3. curtosis of wavelet transformed image
# 4. entropy of image
# 5. class
df.head()
df.describe()
df.info()

#---------------------------------------------------------------
# Data Exploration

sns.countplot(data=df, x='Class')
sns.pairplot(data=df, hue='Class')

#---------------------------------------------------------------
# Data Preparation

# Applying scaler to dataset
# Not necessary for this dataset, but for practice sake
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Class', axis=1))
scaled_features = scaler.fit_transform(df.drop('Class', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()

#---------------------------------------------------------------
# Train Test Split

from sklearn.model_selection import train_test_split
X = df_feat
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)

#---------------------------------------------------------------
#Tensorflow

# Creating a list of feature column objects
df_feat.columns
image_var = tf.feature_column.numeric_column('Image.Var')
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy = tf.feature_column.numeric_column('Entropy')
feat_cols = [image_var, image_skew, image_curt, entropy]


# Create classifier for DNN Classifier
# Dataset contains 2 classes to set in n_classes
# Plug in feature columns object created above in formula
# input_fn takes in X_train and y_train, batch size is determined by data size
# Set steps to learn from data
classifier = tf.estimator.DNNClassifier(hidden_units=[10,20,10], n_classes=2, feature_columns=feat_cols)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=20, shuffle=True)
classifier.train(input_fn=input_func, steps=500)

#---------------------------------------------------------------
# Model Evaluation

# Create another input_fn to take in X_test and use the classifier to predict
# Set shuffle to false since predictions don't need to be shuffled
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)
note_predictions = list(classifier.predict(input_fn=pred_fn))
note_predictions[0]

final_preds = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds))

#---------------------------------------------------------------
# Optional Comparison

# Running these lines, the Random Forest Classifier also performs well
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_preds = rfc.predict(X_test)
print(confusion_matrix(y_test, rfc_preds))
print(classification_report(y_test, rfc_preds))




# Project source: Pierian Data