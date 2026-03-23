# 🚗 AI-Driver Drowsiness Detection System – Project Documentation

---

## 📌 1. Project Overview

The **Driver Drowsiness Detection System** is a Machine Learning-based web application that predicts whether a driver is **Drowsy** or **Alert** based on physiological and behavioral inputs.

The system uses:

* A trained **Deep Learning model (TensorFlow/Keras)**
* A **Flask web application** for deployment
* A **user-friendly HTML interface** for input

---

## 🎯 2. Objective

The main goal of this project is to:

* Detect driver fatigue in real-time (via input features)
* Prevent accidents caused by drowsiness
* Provide a simple interface for prediction

---

## 🏗️ 3. System Architecture

```
User Input (HTML Form)
        ↓
Flask Backend (app.py)
        ↓
Data Preprocessing (Scaler)
        ↓
Deep Learning Model (drowsiness_model.h5)
        ↓
Prediction (Drowsy / Alert)
        ↓
Result Display (HTML)
```

---

## ⚙️ 4. Technologies Used

| Technology       | Purpose                    |
| ---------------- | -------------------------- |
| Python           | Backend logic              |
| Flask            | Web framework              |
| TensorFlow/Keras | Deep learning model        |
| Pandas & NumPy   | Data handling              |
| Pickle           | Loading scaler & threshold |
| HTML/CSS         | Frontend UI                |

---

## 📂 5. Project Files Description

### 5.1 `app.py` (Backend)

This is the main Flask application that:

* Loads the trained model
* Accepts user input
* Performs prediction
* Returns results

#### 🔹 Key Components:

**1. Import Libraries**

```python
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
```

**2. Load Model and Preprocessing Files**

```python
scaler = pickle.load(open("scaler.pkl", "rb"))
threshold = pickle.load(open("threshold.pkl", "rb"))
model = tf.keras.models.load_model("drowsiness_model.h5", compile=False)
```

**3. Home Route**

```python
@app.route('/')
def home():
    return render_template("index.html")
```

**4. Prediction Route**

* Takes input from form
* Converts into DataFrame
* Applies scaling
* Predicts probability
* Compares with threshold

```python
prob = model.predict(df_scaled)[0][0]
prediction = 1 if prob > threshold else 0
```

**5. Output**

```python
result = "Drowsy" if prediction == 1 else "Alert"
```

---

### 5.2 `index.html` (Frontend)

This file provides a **modern UI** for user input.

#### 🔹 Features:

* Glassmorphism design ✨
* Background image with overlay
* Smooth animations
* Responsive input form

#### 🔹 Input Fields:

| Feature                | Description            |
| ---------------------- | ---------------------- |
| Age                    | Driver age             |
| Gender                 | 0 = Male, 1 = Female   |
| Blink Rate             | Eye blinking frequency |
| Eye Closure Duration   | Eye closing time       |
| Yawning Count          | Number of yawns        |
| Heart Rate             | BPM                    |
| Head Tilt Angle        | Head movement          |
| Steering Variation     | Driving stability      |
| Reaction Time          | Driver response speed  |
| Sleep Hours Last Night | Sleep duration         |

---

## 🧠 6. Machine Learning Model

### Model Type:

* Deep Learning (Neural Network)

### Input Features:

10 features including physiological and behavioral data.

### Processing Steps:

1. Input collected from user
2. Converted to DataFrame
3. Scaled using `scaler.pkl`
4. Passed into trained model
5. Output probability generated

### Decision Rule:

```
If probability > threshold → Drowsy
Else → Alert
```

---

## 🔄 7. Data Flow

1. User enters data in web form
2. Data sent via POST request to `/predict`
3. Flask processes data
4. Model predicts probability
5. Result displayed to user

---

## 🚀 8. How to Run the Project

### Step 1: Install Dependencies

```bash
pip install flask pandas numpy tensorflow
```

### Step 2: Ensure Files Exist

* `app.py`
* `index.html`
* `result.html`
* `scaler.pkl`
* `threshold.pkl`
* `drowsiness_model.h5`

### Step 3: Run Flask App

```bash
python app.py
```

### Step 4: Open Browser

```
http://127.0.0.1:5000/
```

---

## 📊 9. Output

The system outputs:

* **Prediction:** Drowsy / Alert
* **Probability Score**

Example:

```
Prediction: Drowsy
Probability: 0.82
```

---

## ⚠️ 10. Limitations

* Depends on input accuracy
* Not real-time (no camera integration)
* Model performance depends on training data
* Requires proper scaling and preprocessing

---

## 🔮 11. Future Improvements

* Real-time detection using webcam 🎥
* Integration with IoT sensors
* Alert system (alarm/vibration)
* Mobile app version 📱
* Higher accuracy model tuning

---

## 12. Conclusion

This project successfully demonstrates:

* End-to-end ML deployment
* Integration of Deep Learning with Flask
* Real-world application in driver safety

It can be further enhanced into a **real-time accident prevention system**.

---



