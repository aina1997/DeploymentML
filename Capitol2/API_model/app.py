
# importing Flask and other modules
from flask import Flask, request, render_template
import joblib
from classes.customTransformers import CustomTransformer
import pandas as pd
import json

 
# Flask constructor
app = Flask(__name__)  
app.run(debug=True)
 
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["POST"])
def titanic():

   
   try:
      data = pd.read_csv('../test.csv', index_col='PassengerId')
      result = []

      filename = '../best_model.pkl'
      
      pipeline_loaded = joblib.load('../best_model.pkl')
      result = pipeline_loaded.predict(data)
      return {'message': True, "result": result.tolist()}


   except:
      # otherwise the userId does not exist
      return {
         'message': "Error"
      }, 404
 
if __name__=='__main__':
   app.run()