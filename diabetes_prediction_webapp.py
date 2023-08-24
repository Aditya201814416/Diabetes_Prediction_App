# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 20:21:52 2023

@author: Aditya
"""

import numpy as np
import pickle 
import streamlit as st

#loading the saved model 
loaded_model = pickle.load((open('C:/Users/Aditya/Desktop/Deploy_Diabetes/trained_model.sav', 'rb')))

#creating a function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array= np.asarray(input_data)

    #reshaping the data because we only want data for one instance


    #standardizing the input data since our model got trained on standardized data and not general raw data
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):    ##prediction is a list . svm returns a list with only one value(0 or 1)
      return "The Person does not have diabetes"
    else:
      return "The person is  diabetic"
    
    
   
   

#creating the streamlit ui function

def main():
    
    
    #giving a title 
    st.title('Diabetes Predictor')
    
    #getting the input data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure= st.text_input('Blood Pressure value')
    SkinThickness= st.text_input('Skin Thickness value')
    Insulin= st.text_input('Insulin Level')
    BMI= st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age= st.text_input('Age of the Person')
    
    #code for prediction
    
    diagnosis =''
    
    #creating an input button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose, BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()
    
         
         
         
         
         