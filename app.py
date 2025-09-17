from flask import Flask, render_template, request
import numpy as np
import pickle
from PIL import Image
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model.pkl")


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    
    img = Image.open(file).convert('L')   
    img = img.resize((28, 28))            
    img_array = np.array(img) / 255.0     

    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array


    img_array = (img_array > 0.5).astype(np.float32)

    img_array = img_array.reshape(1, -1)

    try:
        prediction = model.predict(img_array)

        if prediction.ndim > 1 and prediction.shape[1] > 1:
            predicted_digit = int(np.argmax(prediction[0]))
        else:
            predicted_digit = int(prediction[0])

    except Exception as e:
        return f"Prediction error: {str(e)}"


    return render_template("index.html", prediction=predicted_digit)

if __name__ == '__main__':
    app.run(debug=True)
