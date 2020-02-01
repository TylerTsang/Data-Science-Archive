# -*- coding: utf-8 -*-

#-------------------------------------------------
# Data Overview

# For this project we will be using data from Lending Club
# It is a platform that connects borrowers with investors
# For investors it is important that borrowers pay them back
# Let's see if we predict whether borrowers pay back based on characteristics

# The Data contains the following:

# Credit Policy:             1 for customer meeting credit underwriting criteria, 0 otherwise
# Purpose:                   Purpose for the loan (credit_card, debt_consolidation, educational, major_purchase, small_business, all_other)
# Interest Rate:             Interest rate of the loan, higher risk borrowers have higher interest rates
# Installment:               Monthly installments owed if loan is funded
# Log Annual Income:         The natural log of self-reported annual income of the borrower
# Debt to Income (dti):      The debt to income ratio of the borrower
# FICO:                      The FICO score of the borrower
# Days With Credit Line:     Number of days the borrower has had a credit line
# Revolving Balance:         The borrower's revolving balance (unpaid at the end of the billing cycle)
# Revolving Utility:         The borrower's revolving utilization rate (credit line used relative to total credit available)
# Inquiries Last 6 Months:   The borrower's number of inquiries by creditors in the last 6 months
# Delingquency 2 Years:      Times the borrower had been +30 days past due on a payment in the last 2 years
# Public Records:            The borrower's number of derogatory public records (bankruptcy filings, tax lines, or judgements)

#-------------------------------------------------
# Importing Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#------------------------------------------------
# Getting the Data

loan = pd.read_csv('loan_data.csv')

loan.head()
loan.describe()
loan.info()

#------------------------------------------------
# Data Exploration

# Let's first see the relationship between whether the borrower meeting credit criteria + their fico scores
sns.set_style('darkgrid')
plt.figure(figsize=(10, 6))
loan[loan['credit.policy']==1]['fico'].hist(alpha=0.5, color='blue',
                                            bins=30, label='Credit Policy = 1')
loan[loan['credit.policy']==0]['fico'].hist(alpha=0.5, color='red',
                                            bins=30, label='Credit Policy = 0')
plt.legend()
plt.xlabel('FICO')

# Now let's try and see the relationship on whether the loan has been paid and FICO scores
plt.figure(figsize=(10, 6))
loan[loan['not.fully.paid']==1]['fico'].hist(alpha=0.5, color='red',
                                            bins=30, label='Not Fully Paid = 1')
loan[loan['not.fully.paid']==0]['fico'].hist(alpha=0.5, color='blue',
                                            bins=30, label='Not Fully Paid = 0')
plt.legend()
plt.xlabel('FICO')

# Let's do a countplot and see if purpose had any relationship to loans being paid
plt.figure(figsize=(10, 6))
sns.countplot(data=loan, x='purpose', hue='not.fully.paid', palette='Set1')

# Let's see about FICO scores and interest rates
sns.jointplot(data=loan, x='fico', y='int.rate', color='blue', s=5)

# Lastly, let's try FICO and interest rates as above but add some granularity
# Let's add in whether borrowers meet credit criteria and whether they paid back
plt.figure(figsize=(11,7))
sns.lmplot(data=loan, x='fico', y='int.rate', hue='credit.policy',
           col='not.fully.paid', palette='Set1', scatter_kws={'s':5})

#------------------------------------------------
# Setting up the Data for the model

# Purpose is our only categorical feature in the data set
# Let's make it numerical so a machine can read it
cat_feats = ['purpose']

# Dummies replace categorical features and replace with 0 or 1 to indicate a feature occurs or not
final_data = pd.get_dummies(loan, columns=cat_feats, drop_first=True)

# To see whether categorical features were changed to dummy variables
final_data.info()

#------------------------------------------------
#Train Test Split

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#------------------------------------------------
# Decision Tree Model

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

# It seems like the decision tree performed poorly, 
# Let's see if a random forest will change that 
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

#------------------------------------------------
# Random Forest Model

rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)

# It seems like both our decision tree and random forest performed poorly
# As a result, we should conclude that more feature engineering is needed to improve accuracy
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))





# Project Source: Pierian Data