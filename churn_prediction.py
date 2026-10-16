import pandas as pd                    #Used for reading and manipulating data.
import numpy as np                      #Used for mathematical and numerical operations.
import matplotlib.pyplot as plt                   #Used to create charts and graphs.
import seaborn as sns                              #Used for advanced and attractive visualizations.
from sklearn.model_selection import train_test_split             #Splits data into training and testing sets.
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder    # MinMaxScaler- Scales values between 0 and 1. & OneHotEncoder- Converts categorical columns into numerical values.
from sklearn.metrics import classification_report                #Evaluates model performance.
from sklearn.utils.class_weight import compute_class_weight            #Class Weights- Handles imbalanced datasets.
import tensorflow as tf                        #Deep Learning library used to build Neural Networks.
from keras import Sequential, layers           #Sequential- Creates a neural network layer by layer. & Layers- Adds neural network layers.


df = pd.read_csv('D:/OneDrive - Encore Asset Reconstruction Company Pvt Ltd/Downloads/Churn_Modelling.csv')        #pd = pandas library, read_csv() = reads a CSV file, df = DataFrame (a table of rows and columns)
df.sample(5)          #sample() returns random rows from the dataset. & 5 means return 5 random records.


df.info()      #df.info() gives a summary of the DataFrame.

df.drop(columns= ['RowNumber','CustomerId', 'Surname'] ,axis= 1, inplace= True) 

df.dtypes
