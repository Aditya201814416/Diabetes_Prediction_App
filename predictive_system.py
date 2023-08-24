# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle


#loading the saved model
loaded_model = pickle.load((open('C:/Users/Aditya/Desktop/Deploy_Diabetes/trained_model.sav', 'rb')))

input_data =(10,168,74,0,0,38,0.537,34)
input_data_as_numpy_array= np.asarray(input_data)

#reshaping the data because we only want data for one instance


#standardizing the input data since our model got trained on standardized data and not general raw data
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):    ##prediction is a list . svm returns a list with only one value(0 or 1)
  print("The Person does not have diabetes")
else:
  print("The person is  diabetic")