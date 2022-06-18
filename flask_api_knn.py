# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 23:51:50 2022

@author: pratiksanghvi
"""
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
Swagger(app)

pickle_in = open("knn_regressor.pkl","rb")
regressor=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template('index.html')
#    return render_template('C:\\Users\\prati\\Internship_iNeuron\\Coding\\index.html')

   
@app.route('/predict',methods=["Get"])
def predict_get():
    
   
    age = request.args.get("age")
    bmi = request.args.get("bmi")
    sex = request.args.get("sex")
    children = request.args.get("children")
    smoker = request.args.get("smoker")
    region = request.args.get("region")

    #====convert to dummies data frame for the user input
    #Prepare the dataset for the model to predict
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

    return df

@app.route('/predict_post',methods=["POST"])
def predict():
    
    features=[]
    for x in request.form.values():
        features.append(x)
        
    age = features[0]
    bmi = features[1]
    sex = features[2]
    children = features[3]
    smoker = features[4]
    region = features[5]  
    print(features)
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
    print(df)

    prediction=regressor.predict(df)
    print(prediction)
            
    return render_template('index.html', 
                          prediction_text='Employee Salary should be $ {}'.format(prediction))


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)
    
    