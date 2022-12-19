import os
import psycopg2
from dotenv import load_dotenv
from flask import Flask, request, render_template
from flask import jsonify
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

@app.route("/api/titanic", methods =["GET", "POST"], endpoint="titanic")
def get_prediction():

    ## If post
    if request.method == "POST":

        columns = ['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
        values = []

        ## Check columns
        for column in columns:
            
            value = request.form.get(column)
            if value == "":
                value = None
            values.append(value)

        ## Try predict
        #try:
        print(values)
        df= pd.DataFrame(np.array([values]), columns=columns)
        result = []

        filename = './best_model.pkl'
            
        pipeline_loaded = joblib.load('./best_model.pkl')
        result = pipeline_loaded.predict(df)
            
            ## Resposta
        response = app.response_class(
                    response=json.dumps({'message': True, "result": result.tolist()}),
                    status=201,
                    mimetype='application/json'
            )
        return response


        #except:
        # otherwise the userId does not exist
            # response = app.response_class(
            #     response=json.dumps({'error': "Server error"}),
            #     status=404,
            #     mimetype='application/json'
            # )
                
            # return response

    return render_template("form.html")



    

