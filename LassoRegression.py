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


# In[4]:


# Store the list of column names 
colnames = list(insuranceData.columns)
colnames


# In[ ]:


insuranceData['smoker'] = insuranceData['smoker'].replace({"yes":1,"no":0})
insuranceData['sex'] = insuranceData['sex'].replace({"male":1,"female":0})


# In[ ]:


# Smoker vs Expenses almost 79% correlation
# Age vs Expenses almost 30% correlation
# BMI vs Expenses almost 20% correlation
# Number of children ,region ,sex are not contributing to expenses


# In[5]:


insuranceData['smoker'] = insuranceData['smoker'].replace({"yes":1,"no":0})
insuranceData['sex'] = insuranceData['sex'].replace({"male":1,"female":0})
insuranceData['region'] = insuranceData['sex'].replace({"southwest":0,"southeast":1,"northeast":2,"northwest":3})

#insuranceData.corr()
insuranceData.info()


# In[32]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression  #Import Linear regression model
from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV
from numpy import mean
from numpy import std
from numpy import absolute
from numpy import arange

y=insuranceData['expenses']
X=insuranceData.drop(columns='expenses')
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.25,random_state=0)


# In[24]:


lasso_model = Lasso(alpha=1.0)
lasso_model.fit(train_X,train_y)


# In[25]:


print("Intercept : ", lasso_model.intercept_)
print("Slope : ", lasso_model.coef_)


# In[26]:


# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
# evaluate model
scores = cross_val_score(lasso_model, train_X, train_y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
print("====================================")


# In[27]:


#Predicting TEST & TRAIN DATA
train_predict = lasso_model.predict(train_X)
test_predict = lasso_model.predict(test_X)


# In[29]:


print("====================================")
print("MAE")
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


# In[44]:


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


# In[35]:


lasso_model = Lasso()
# define model evaluation method
cv = RepeatedKFold(n_splits=100, n_repeats=20, random_state=1)
# define grid
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)

# define search
search = GridSearchCV(lasso_model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# perform the search
results = search.fit(train_X,train_y)

# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)


# In[37]:


#Predicting TEST & TRAIN DATA
train_predict = results.predict(train_X)
test_predict = results.predict(test_X)


# In[40]:


from sklearn.linear_model import LassoCV

# define model
lasso_model_CV = LassoCV(alphas=arange(0, 1, 0.01), n_jobs=-1)
lasso_model_CV.fit(train_X,train_y)

#print("Intercept : ", lasso_model.intercept_)
#print("Slope : ", lasso_model.coef_)

#Predicting TEST & TRAIN DATA
train_predict_CV = lasso_model_CV.predict(train_X)
test_predict_CV = lasso_model_CV.predict(test_X)
print("====================================")
print("MAE")
print("Train : ",mean_absolute_error(train_y,train_predict_CV))
print("Test  : ",mean_absolute_error(test_y,test_predict_CV))
print("====================================")

print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict_CV))
print("Test  : ",mean_squared_error(test_y,test_predict_CV))
print("====================================")

print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict_CV)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict_CV)))
print("====================================")

print("R2 SCORE")
print("Train : ",r2_score(train_y,train_predict_CV))
print("Test  : ",r2_score(test_y,test_predict_CV))  
print("====================================")


# In[43]:


plt.figure(figsize=(10,10))
plt.scatter(test_y,test_predict_CV, c='crimson')
#plt.yscale('log')
#plt.xscale('log')

p1 = max(max(test_predict_CV), max(test_y))
p2 = min(min(test_predict_CV), min(test_y))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[ ]:




