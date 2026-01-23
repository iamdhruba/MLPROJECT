from flask import Flask, request, render_template
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

#route
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template("student.html")
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score')
        )

        #convert data to DataFrame
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        print('Before Prediction')
        predict_pipeline = PredictPipeline()
        print('Mid Phase of prediction')
        result = predict_pipeline.predict(pred_df)
        print('After Prediction')
        print(result)
        return render_template("student.html", results = result[0])
    

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 8080)