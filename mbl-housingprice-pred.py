#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:36:36 2018

@author: mlbook

Submission format @ Kaggle, where train data and test data are downloaded

Level 1

Using RandomTreeForest model

"""
#Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#Read the data into dataframe        [[Training Data from train.csv file]]
# if data is from web portal           ../input/train.csv
dataset = pd.read_csv('train.csv')
train_y = dataset.SalePrice
#dataset.columns   'for selecting the intended variables/columns 
need_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = dataset[need_cols]
#Fitting into the model
rfg_model=RandomForestRegressor()
rfg_model.fit(train_X, train_y)


#Read the data into dataframe        [[Training Data from test.csv file without SalePrice data]]
dataset1 = pd.read_csv('test.csv')
test_X = dataset1[need_cols]

#Predicting the training values
sp_pred = rfg_model.predict(test_X)
print(sp_pred)

#Preparing for submission
my_submit = pd.DataFrame({'ID':dataset1.Id, 'SalePrice':sp_pred})

#Exporting to CSV file
my_submit.to_csv('mysubmission.csv',index=False)
















