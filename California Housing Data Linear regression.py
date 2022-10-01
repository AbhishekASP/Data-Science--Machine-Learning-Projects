#!/usr/bin/env python
# coding: utf-8

# # California Housing Data Modeling

# In[ ]:


# Importing the packages needed for Linear Regression Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.rc('figure',figsize=(12,6))

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# # Data Importing

# In[ ]:




data =pd.read_csv('C:\\Users\\User\\Desktop\\Masters in Computer Science\\Data Science\\Projects\\housing.csv')


# In[6]:


data.head(10)


# # Data Cleaning
# 
#     # 1. Missing Values 
#     # 2. Outliers

# In[ ]:




data.isnull().sum()


# In[8]:


# Fill missing values in the column with Mean value.

missing_column =['total_bedrooms']

for i in missing_column:
    data.loc[data.loc[:,i].isnull(),i] = data.loc[:,i].mean()


# In[9]:


# See outliers at a high level

data.describe()


# In[10]:


# Detect outliers using functions

def outliers(mydata):
    dataoutlier = []
    mydata = sorted(mydata)
    
    q1 = np.percentile(mydata,0.25)
    q3 = np.percentile(mydata,0.75)
    IQR = q3-q1
    outlier_low = q1-(1.5*IQR)
    outlier_upp = q3+(1.5*IQR)
    
    for i in mydata:
        if(i < outlier_low or i > outlier_upp):
            dataoutlier.append(i)
    return dataoutlier    


# In[68]:


#outliers(data['total_rooms'])


# In[12]:


# Population Max -35682.0 
# Population Min -3
# Notting both these values to know the outlier treatment later.


# In[15]:


data.columns


# In[18]:


# Outlier treament of all columns.
# replace outlier with 10th and 90th values

columns =['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value']

for i in columns:
    ninetieth_percentile = np.percentile(data[i],90) # data[i] is one particular column
    tenth_percentile = np.percentile(data[i],10)
    data[i] = np.where(data[i] > ninetieth_percentile , ninetieth_percentile, data[i])#[np.where(condition,x,y)if True x,else y}
    data[i] = np.where(data[i] < tenth_percentile , tenth_percentile, data[i])


# In[19]:


data.describe() # here we can see population outliers removed.

# Thus data Cleaning is done Completely.


# # Data Exploration

# In[21]:


data.corr() # Shows relation btw 2 variables.
# values greater than 0.8 shows strong correlation.


# In[25]:


# Check prices of houses near the ocean

sns.boxplot(x='ocean_proximity',y='median_house_value',data=data)
#island houses have higher prices.


# In[26]:


# Near Bay and Island houses last longer.
sns.boxplot(x='ocean_proximity',y='housing_median_age',data=data)


# In[30]:


sns.barplot(x='ocean_proximity',y='median_income',data=data)


# # Feature Engineering

# In[33]:


data['is_island'] = data.apply(lambda x : 1 if (x['ocean_proximity']=='ISLAND') else 0, axis=1)


# In[34]:


data.head(10)


# # Model Training

# In[40]:


dummified = pd.get_dummies(data,columns=['ocean_proximity'])

# to change categorical columns into numerical values by splitting


# In[41]:


dummified.head(10)


# In[42]:


dummified.columns


# In[43]:


# Linear regression equation y=mx1+mx2+mx3+mx4....+c

# Giving all other columns we need to predict the median house value


# In[51]:


x = dummified[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
        'is_island', 'ocean_proximity_<1H OCEAN',
       'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
       'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']]

y = dummified[['median_house_value']]


# In[53]:


# here we want to split the data for training(larger data set) and testing(lesser data set) from the data set we have
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.20)


# # Running linear regression

# In[55]:


# Train my model
model = LinearRegression()
model.fit(xtrain,ytrain) ### these both go to the model


# In[64]:


### This is the available model now
### Now predict with the test data

x_pred = model.predict(xtest)  # predict xtest based on the model(xtrain,ytrain)


# In[61]:


model.coef_


# In[62]:


model.intercept_  ### the slope value C


# # Model Evaluation
# 
#     # Now compare xtest with ytest

# In[66]:


r2_score(ytest,x_pred) ### How good is your model


# # Model is 65% accurate.
