# -*- coding: utf-8 -*-

# Objective: Explore whether the company should focus on its mobile app or website

#---------------------------------------------------
# Importing libraries 

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#---------------------------------------------------
# Getting the data

customers = pd.read_csv('Ecommerce Customers')
customers.head()
customers.describe()
customers.info()

#---------------------------------------------------
# Data Exploration

sns.set_palette('GnBu_d')
sns.set_style('whitegrid')

sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')

# Time on app seems to have a slight correlation to yearly amount spent.
sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent')
sns.jointplot(data=customers, x='Time on App', y='Length of Membership', kind='hex')

# Viewing all the plots, we see a clearn relationship between Length of membership and yearly amount spent
# Let's use this in a lineplot to see
sns.pairplot(data=customers)

sns.lmplot(data=customers, x='Length of Membership', y='Yearly Amount Spent')

#--------------------------------------------------
# Train Test Split

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#--------------------------------------------------
# Training the Model

lm = LinearRegression()
lm.fit(X_train, y_train)

# Coefficients:
print('Coefficients: \n', lm.coef_)

#---------------------------------------------------
# Predicting with the Test Data

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel('Y test')
plt.ylabel('Predicted Y')

#----------------------------------------------------
# Model Evaluation

# Printing error metrics:
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Residuals:
sns.distplot((y_test-predictions), bins=60);

#----------------------------------------------------
# Conclusion

coefficients = pd.DataFrame(lm.coef_, X.columns)
coefficients.columns = ['Coefficient']
coefficients

# 1 unit increase of Avg. Session Length = 25.72 increase in total dollars spent
# 1 unit increase of Time on App = 38.60 increase in total dollars spent
# 1 unit increase of Time on Website = 0.46 increase in total dollars spent
# 1 unit increase of Length of Membership = 61.67 increase in total dollars spent

# Should the company focus on the mobile app or website?

# The website could be further developed to catch up to the app
# The app generates more, therefore it might be best to focus on it
# It depends on other factors in the company as well
# Might explore the relationship between Length of Membership and the app or website before concluding





