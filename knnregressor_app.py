# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 23:04:19 2022

@author: pratiksanghvi
"""
from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
from flasgger import Swagger
import streamlit as st 
import str
from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("knn_regressor.pkl","rb")
regressor=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_expenses_knn(age,bmi,sex,children,smoker,region):
    
    #Prepare the dataset for the model to predict
    df = pd.DataFrame({"age":age,"bmi":bmi,"sex_0":0,"sex_1":0,"children_0":0,
                       "children_1":0,"children_2":0,"children_3":0,
                       "children_4":0,"children_5":0,
                       "smoker_0":0,"smoker_1":0,
                       "region_0":0,"region_1":0,"region_2":0,"region_3":0})    
   
    if(sex.lower()=="male"):
        df['sex_1']=df['sex'].replace([0],1)
    elif(sex.lower()=="female"):
        df['sex_0']=df['sex'].replace([0],1)

    
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
        df['smoker_1']=df['smoker_1'].replace([0],1)
    elif(smoker.lower=="no"):
        df['smoker_0']=df['smoker_0'].replace([0],1)       
    
    if(region.replace(" ","").lower=="northeast"):
        df['region_0']=df['region_0'].replace([0],1)
    elif(region.replace(" ","").lower=="northwest"):
        df['region_1']=df['region_1'].replace([0],1)
    elif(region.replace(" ","").lower=="southeast"):
        df['region_2']=df['region_2'].replace([0],1)
    elif(region.replace(" ","").lower=="southwest"):
        df['region_3']=df['region_3'].replace([0],1)
    
    
    
    
    
    prediction=regressor.predict(df)
    print(prediction)
    return prediction



def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input("age","Type Here")
    bmi = st.number_input("bmi","Type Here")
    sex = st.text_input("sex","Type Here")
    children = st.number_input("chilren","Type Here",start=0,end=5)
    smoker = st.text_input("smoker","Type Here")
    region = st.text_input("region","Type Here")

    result=""
    if st.button("Predict"):
        result=predict_expenses_knn(age,bmi,sex,children,smoker,region)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    
