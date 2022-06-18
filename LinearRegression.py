#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[3]:


# Import the data into the system
insuranceData = pd.read_csv("C:\\Users\\prati\\Internship_iNeuron\\Dataset\\insurance.csv")
insuranceData.head()


# In[33]:


# Store the list of column names 
colnames = list(insuranceData.columns)
colnames



# In[9]:


insuranceData['smoker'] = insuranceData['smoker'].replace({"yes":1,"no":0})
insuranceData['sex'] = insuranceData['sex'].replace({"male":1,"female":0})
insuranceData['region'] = insuranceData['sex'].replace({"southwest":0,"southeast":1,"northeast":2,"northwest":3})

#insuranceData.corr()
insuranceData.info()


# In[4]:


# Smoker vs Expenses almost 79% correlation
# Age vs Expenses almost 30% correlation
# BMI vs Expenses almost 20% correlation
# Number of children ,region ,sex are not contributing to expenses


# In[25]:

#['age', 'sex', 'bmi', 'children', 'smoker', 'region']
columns_new = list(['sex', 'children', 'smoker', 'region'])
categorical_cols = list(insuranceData[['sex', 'children', 'smoker', 'region']])
encodedDf_insurance = pd.get_dummies(insuranceData, columns = categorical_cols)
encodedDf_insurance.head()

# In[10]:


df1 = insuranceData[list(insuranceData.columns)[0:5]]


# In[11]:


from sklearn.linear_model import LinearRegression  #Import Linear regression model
from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
train_X,test_X,train_y,test_y=train_test_split(df1,insuranceData['expenses'],test_size=0.3,random_state=0)


# In[12]:


model = LinearRegression()
model.fit(train_X,train_y)


# In[13]:


train_predict = model.predict(train_X)
test_predict = model.predict(test_X)


# In[14]:


print("R2 SCORE")
print("Train : ",r2_score(train_y,train_predict))
print("Test  : ",r2_score(test_y,test_predict))  
print("====================================")


# In[16]:


plt.figure(figsize=(10,10))
plt.scatter(test_y,test_predict, c='crimson')
#plt.yscale('log')
#plt.xscale('log')

p1 = max(max(test_predict), max(test_y))
p2 = min(min(test_predict), min(test_y))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[19]:


df = pd.DataFrame()

df['Expected']  = pd.Series(test_y)
df['Predicted'] = pd.Series(test_predict)

figure = plt.figure(figsize=(15, 10))

axes = sns.scatterplot(data=df, x='Expected', y='Predicted', 
                       hue='Predicted', palette='cool', 
                       legend=True)

start = min(test_y.min(), test_predict.min())
end   = max(test_y.max(), test_predict.max())

axes.set_xlim(start, end)
axes.set_ylim(start, end)

line = plt.plot([start, end], [start, end], 'k--')

