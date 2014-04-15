import numpy as np
import pandas as pd
import statsmodels.api as sm

file_path = "C:/Users/muraki/Dropbox/Mari/courses/Udacity/Intro to DS/projects/titanic_data.csv"

def complex_heuristic(file_path):
    '''
    https://www.udacity.com/course/ud359
        
    You are given a list of Titantic passengers and their associating
    information. More information about the data can be seen at the link below:
    http://www.kaggle.com/c/titanic-gettingStarted/data

    For this exercise, you need to write a more sophisticated heuristic
    that will use the passengers' gender and their social economical class and age 
    to predict if they survived the Titanic diaster. 
    
    You prediction should be 79% accurate or higher.
    
    If the passenger is female or if his/her socio-economical status is high AND
    if the passenger is under 18, you should assume the passenger survived.
    Otherwise, you should assume the passenger perished in the disaster.
    
    Or more specifically in code terms: female or (high status and under 18)
    
    You can access the gender of a passenger via passenger['Sex'].
    If the passenger is male, passenger['Sex'] will return a string "male".
    If the passenger is female, passenger['Sex'] will return a string "female".
    
    You can access the socio-economical status of a passenger via passenger['Pclass']:
    High socio-economical status -- passenger['Pclass'] is 1
    Medium socio-economical status -- passenger['Pclass'] is 2
    Low socio-economical status -- passenger['Pclass'] is 3

    You can access the age of a passenger via passenger['Age'].
    
    Write your prediction back into the "predictions" dictionary. The
    key of the dictionary should be the Passenger's id (which can be accessed
    via passenger["PassengerId"]) and the associating value should be 1 if the
    passenger survived or 0 otherwise. 

    For example, if a passenger survived:
    passenger_id = passenger['PassengerId']
    predictions[passenger_id] = 1

    Or if a passenger perished in the disaster:
    passenger_id = passenger['PassengerId']
    predictions[passenger_id] = 0
    
    You can also look at the titantic data that you will be working with
    at the link below:
    https://www.dropbox.com/s/r5f9aos8p9ri9sa/titanic_data.csv
    '''

    df = pd.read_csv(file_path)
    predictions = {}
    for passenger_index, passenger in df.iterrows():
        if passenger['Sex'] == 'female' or (passenger['Pclass'] == 1 and passenger['Age'] < 18):
            predictions[passenger['PassengerId']] = 1
        else:
            predictions[passenger['PassengerId']] = 0
    return predictions

    