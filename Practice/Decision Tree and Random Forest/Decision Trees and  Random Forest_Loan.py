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

import numpy as np
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

# Now let's try and see the relationship on whether the loan has been paid and fico scores
plt.figure(figsize=(10, 6))
loan[loan['not.fully.paid']==1]['fico'].hist(alpha=0.5, color='red',
                                            bins=30, label='Not Fully Paid = 1')
loan[loan['not.fully.paid']==0]['fico'].hist(alpha=0.5, color='blue',
                                            bins=30, label='Not Fully Paid = 0')
plt.legend()
plt.xlabel('FICO')

#
plt.figure(figsize=(10, 6))
sns.countplot(data=loan, x='purpose', hue='not.fully.paid', palette='Set1')

#
sns.jointplot(data=loan, x='fico', y='int.rate', color='blue', s=5)

#
plt.figure(figsize=(11,7))
sns.lmplot(data=loan, x='fico', y='int.rate', hue='credit.policy',
           col='not.fully.paid', palette='Set1', scatter_kws={'s':5})

#------------------------------------------------
















