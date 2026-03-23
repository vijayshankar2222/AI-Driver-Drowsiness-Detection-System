from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load model files
scaler = pickle.load(open("scaler.pkl", "rb"))
threshold = pickle.load(open("threshold.pkl", "rb"))
model = tf.keras.models.load_model("drowsiness_model.h5", compile=False)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['Age']),
        float(request.form['Gender']),
        float(request.form['Blink_Rate']),
        float(request.form['Eye_Closure_Duration']),
        float(request.form['Yawning_Count']),
        float(request.form['Heart_Rate']),
        float(request.form['Head_Tilt_Angle']),
        float(request.form['Steering_Variation']),
        float(request.form['Reaction_Time']),
        float(request.form['Sleep_Hours_Last_Night'])
    ]

    df = pd.DataFrame([data], columns=[
        'Age','Gender','Blink_Rate','Eye_Closure_Duration','Yawning_Count',
        'Heart_Rate','Head_Tilt_Angle','Steering_Variation',
        'Reaction_Time','Sleep_Hours_Last_Night'
    ])

    df_scaled = scaler.transform(df)
    prob = model.predict(df_scaled)[0][0]
    prediction = 1 if prob > threshold else 0

    result = "Drowsy" if prediction == 1 else "Alert"

    return render_template("result.html", prediction=result, probability=prob)


if __name__ == "__main__":
    app.run(debug=True)