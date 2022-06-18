#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


# Import the data into the system
insuranceData = pd.read_csv("C:\\Users\\prati\\Internship_iNeuron\\Dataset\\insurance.csv")
insuranceData.head()


# In[19]:


# In[20]:


# Store the list of column names 
colnames = list(insuranceData.columns)
colnames[0:6]


# In[3]:


insuranceData['smoker'] = insuranceData['smoker'].replace({"yes":1,"no":0})
insuranceData['sex'] = insuranceData['sex'].replace({"male":1,"female":0})
insuranceData['region'] = insuranceData['sex'].replace({"southwest":0,"southeast":1,"northeast":2,"northwest":3})

insuranceData.corr()


# In[6]:


# Smoker vs Expenses almost 79% correlation
# Age vs Expenses almost 30% correlation
# BMI vs Expenses almost 20% correlation
# Number of children ,region ,sex are not contributing to expenses
insuranceData.info()


# In[9]:


#import pandas as pd
#['age', 'sex', 'bmi', 'children', 'smoker', 'region']
columns_new = list(['sex', 'children', 'smoker', 'region'])
categorical_cols = list(insuranceData[['sex', 'children', 'smoker', 'region']])
encodedDf_insurance = pd.get_dummies(insuranceData, columns = categorical_cols)
encodedDf_insurance.head()


# In[11]:


y=encodedDf_insurance['expenses']
X=encodedDf_insurance.drop(columns='expenses')


# In[12]:


from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=0)


# In[13]:


# Instantiate DTRegressor model
from sklearn.ensemble import RandomForestRegressor
model_RFReg = RandomForestRegressor(random_state=42)


# In[14]:


model_RFReg.fit(train_X,train_y)


# In[15]:


train_predict = model_RFReg.predict(train_X)
test_predict = model_RFReg.predict(test_X)


# In[16]:


predicted_df = pd.DataFrame({"Actual":test_y,"Predicted":test_predict})
predicted_df.head()


# In[17]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
model_RFReg.score


# In[18]:


print("Train : ",mean_absolute_error(train_y,train_predict))
print("Test  : ",mean_absolute_error(test_y,test_predict))
print("====================================")

print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict))
print("Test  : ",mean_squared_error(test_y,test_predict))
print("====================================")

print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))
print("====================================")

print("R2 SCORE")
print("Train : ",r2_score(train_y,train_predict))
print("Test  : ",r2_score(test_y,test_predict))  
print("====================================")


# In[19]:


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



    