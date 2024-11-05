from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
try:
    model = tf.keras.models.load_model('/home/ec2-user/Flask/model_folder/leaf_disease.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the class indices (update this based on your actual classes)
class_indices = {
    0: "Apple Scab",
    1: "Apple Black Rot",
    2: "Apple Cedar Apple Rust",
    3: "Apple Healthy",
    4: "Cherry Powdery Mildew",
    5: "Cherry Healthy",
    6: "Corn Cercospora Leaf Spot Gray Leaf Spot",
    7: "Corn Common Rust",
    8: "Corn Northern Leaf Blight",
    9: "Corn Healthy",
    10: "Grape Black Rot",
    11: "Grape Esca",
    12: "Grape Leaf Blight",
    13: "Grape Healthy",
    14: "Peach Bacterial Spot",
    15: "Peach Healthy",
    16: "Pepper Bell Bacterial Spot",
    17: "Pepper Bell Healthy",
    18: "Potato Early Blight",
    19: "Potato Late Blight",
    20: "Potato Healthy",
    21: "Strawberry Leaf Scorch",
    22: "Strawberry Healthy",
    23: "Tomato Bacterial Spot",
    24: "Tomato Early Blight",
    25: "Tomato Late Blight",
    26: "Tomato Leaf Mold",
    27: "Tomato Septoria Leaf Spot",
    28: "Tomato Spider Mites",
    29: "Tomato Target Spot",
    30: "Tomato Yellow Leaf Curl Virus",
    31: "Tomato Mosaic Virus",
    32: "Tomato Healthy"
}

def predict_disease(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        print(f"Image shape after preprocessing: {img_array.shape}")
        
        prediction = model.predict(img_array)
        print(f"Raw model prediction: {prediction}")

        predicted_class = np.argmax(prediction, axis=1)[0]
        disease_name = class_indices[predicted_class]
        return disease_name
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Prediction Error"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html', _anchor='about')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', _anchor='predict', result='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', _anchor='predict', result='No selected file')
        if file:
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            disease_name = predict_disease(file_path)
            os.remove(file_path)  # Remove the file after prediction
            return render_template('index.html', _anchor='predict', result=disease_name)
    return render_template('index.html', _anchor='predict')

@app.route('/contact')
def contact():
    return render_template('index.html', _anchor='contact')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host="0.0.0.0", port=3000)
