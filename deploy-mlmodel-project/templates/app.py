from flask import Flask, render_template,request
import pickle
import sklearn
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    gender = int(request.form["gender"])
    age = float(request.form["age"])
    hypertension = int(request.form["hypertension"])
    heart_disease = int(request.form["heart_disease"])
    ever_married = int(request.form["ever_married"])
    work_type = int(request.form["work_type"])
    residence_type = int(request.form["residence_type"])
    avg_glucose_level = float(request.form["avg_glucose_level"])
    bmi = float(request.form["bmi"])
    smoking_status = int(request.form["smoking_status"])
    prediction = model.predict_proba([[gender,age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,bmi,smoking_status]])  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = round(prediction[0][1], 2) 

    return render_template('home.html', prediction_text=f'A person with the given details and has a probability of stroke of {output}.The percentage of getting stroke is {output*100}')

if __name__ == "__main__":
    app.run(debug = True)
