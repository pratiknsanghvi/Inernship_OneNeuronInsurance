#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# Import the data into the system
insuranceData = pd.read_csv("C:\\Users\\prati\\Internship_iNeuron\\Dataset\\insurance.csv")
insuranceData.head()


# In[2]:


#import pandas as pd
#['age', 'sex', 'bmi', 'children', 'smoker', 'region']
columns_new = list(['sex', 'children', 'smoker', 'region'])
categorical_cols = list(insuranceData[['sex', 'children', 'smoker', 'region']])
encodedDf_insurance = pd.get_dummies(insuranceData, columns = categorical_cols)
encodedDf_insurance.head()


# In[3]:


y=encodedDf_insurance['expenses']
X=encodedDf_insurance.drop(columns='expenses')

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=0)


# In[4]:


model_KNNReg = KNeighborsRegressor()
model_KNNReg.fit(train_X,train_y)

train_predict_knn = model_KNNReg.predict(train_X)
test_predict_knn = model_KNNReg.predict(test_X)


# In[5]:


predicted_df = pd.DataFrame({"Actual":test_y,"Predicted":test_predict_knn})
predicted_df.head()


# In[6]:


print("Train : ",mean_absolute_error(train_y,train_predict_knn))
print("Test  : ",mean_absolute_error(test_y,test_predict_knn))
print("====================================")

print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict_knn))
print("Test  : ",mean_squared_error(test_y,test_predict_knn))
print("====================================")

print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict_knn)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict_knn)))
print("====================================")

print("R2 SCORE")
print("Train : ",r2_score(train_y,train_predict_knn))
print("Test  : ",r2_score(test_y,test_predict_knn))  
print("====================================")


# In[ ]:

# Create a pkl file
#import pickle
#pickle_out = open("knn_regressor.pkl","wb")
#pickle.dump(model_KNNReg, pickle_out)
#pickle_out.close()

#import numpy as np
#model_KNNReg.predict([[20,177,0,1,0,1,0,0,0,0,0,1,0,1,0,0]])



s=["22","122","Male","2","Yes","North East"]
features=[]
for x in s:
    features.append(x)   
    
age = features[0]
bmi = features[1]
sex = features[2]
children = features[3]
smoker = features[4]
region = features[5]  
#===================================================
#====convert to dummies data frame for the user input
#Prepare the dataset for the model to predict
df = pd.DataFrame({"age":age,"bmi":bmi,"sex_female":0,"sex_male":0,"children_0":0,
                   "children_1":0,"children_2":0,"children_3":0,
                   "children_4":0,"children_5":0,
                   "smoker_no":0,"smoker_yes":0,
                   "region_northeast":0,"region_northwest":0,"region_southeast":0,"region_southwest":0}, index=[0])     

if(sex.lower()=="male"):
    df['sex_male']=df['sex_male'].replace([0],1)
elif(sex.lower()=="female"):
    df['sex_female']=df['sex_female'].replace([0],1)
    
if(children==0):
    df['children_0']=df['children_0'].replace([0],1)
elif(children==1):
    df['children_1']=df['children_1'].replace([0],1)
elif(children==2):
    df['children_2']=df['children_2'].replace([0],1)
elif(children==3):
    df['children_3']=df['children_3'].replace([0],1)
elif(children==4):
    df['children_4']=df['children_4'].replace([0],1)
elif(children==5):
    df['children_5']=df['children_5'].replace([0],1)

if(smoker.lower=="yes"):
    df['smoker_yes']=df['smoker_yes'].replace([0],1)
elif(smoker.lower=="no"):
    df['smoker_no']=df['smoker_no'].replace([0],1)       

"""if(region.replace(" ","").lower=="northeast"):
    df['region_northeast']=df['region_northeast'].replace([0],1)
elif(region.replace(" ","").lower=="northwest"):
    df['region_northwest']=df['region_northwest'].replace([0],1)
elif(region.replace(" ","").lower=="southeast"):
    df['region_southeast']=df['region_southeast'].replace([0],1)
elif(region.replace(" ","").lower=="southwest"):
    df['region_southwest']=df['region_southwest'].replace([0],1)
"""
 #==========================================================


prediction=model_KNNReg.predict(df)
print(prediction)



















