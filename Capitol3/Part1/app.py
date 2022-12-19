import os
import psycopg2
from dotenv import load_dotenv
from flask import Flask, request
from datetime import datetime, timezone
import joblib
from classes.customTransformers import CustomTransformer
import pandas as pd
import json
import numpy as np


load_dotenv()  # loads variables from .env file into environment

app = Flask(__name__)
url = os.environ.get("DATABASE_URL")  # gets variables from environment
connection = psycopg2.connect(url)

@app.post("/api/titanic")
def get_prediction():

    ## PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    ## 892,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q
    ## values = [None,None,'','male',None,None,None,None,None,None, None]
    
    data = request.get_json()

    columns = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    values = []
    for column in columns:
        value = data[column] if column in data else None
        if column in ['Name', 'Sex'] and value == None:
            print(value)
            return {'message': "You must add a name and a sex"}, 404
        values.append(value)

    try:
        df= pd.DataFrame(np.array([values]), columns=columns)
        result = []

        filename = './best_model.pkl'
        
        pipeline_loaded = joblib.load('./best_model.pkl')
        result = pipeline_loaded.predict(df)
        return {'message': True, "result": result.tolist()},201


    except:
      # otherwise the userId does not exist
        return {
         'message': "Error"
        }, 404


    

