# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:51:48 2022

@author: pratiksanghvi
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
#-------------------------------------------
# Import the data into the system
insuranceData = pd.read_csv("D:\\Learn & Projects\\Internship_iNeuron\\Dataset\\insurance.csv")

# Describe the data
insuranceData.describe()
# Inference Total rows = 1338:
# Average age in the dataset is around 39 years   
# Avg BMI is ~30 and everyone has atleast 1 child
# Avg expenses is 13270

# Exploratory Data Analysis
insuranceData[["sex", "age"]].groupby("sex").mean()
# Female - avg age = 39
# Male  - avg age - 38

















