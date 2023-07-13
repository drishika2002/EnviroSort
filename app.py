from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import json
import base64
from io import BytesIO

app = Flask(__name__)

# Load the waste classification model and labels
model = load_model('static/Resources/Model/keras_model.h5')
labels = ["White glass", "Green glass", "Brown glass", "Food items", "Plastic", "Trash", "Metal", "Paper", "Cardboard", "Battery", "Shoes", "Clothes"]

# Define the paths and mappings for waste bins
bins_folder = 'static/Resources/bins'
bin_mapping = {
    0: 'glass_bin.png',
    1: 'glass_bin.png',
    2: 'glass_bin.png',
    5: 'plastic_bin.png',
    6: 'metal_bin.png',
    3: 'organic_bin.png',
    7: 'paper_bin.png',
    8: 'paper_bin.png',
    4: 'plastic_bin.png',
    9: 'battery_bin.png',
    10: 'clothes_shoes_donate.png',
    11: 'clothes_shoes_donate.png'
}

def get_recycling_data():
    with open('recycling_data.json') as file:
        recycling_data = json.load(file)
    return recycling_data

def process_image(image):
    img = cv2.resize(image, (224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label_id = np.argmax(prediction)

    waste_category = labels[label_id]
    confidence = prediction[0][label_id] * 100

    bin_image = bin_mapping.get(label_id, 'default_bin.png')
    bin_image_path = os.path.join(bins_folder, bin_image)

    return waste_category, confidence, bin_image_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def homePage():
    return render_template('home.html')

@app.route('/process', methods=['POST', 'GET'])
def process():
    if request.method == 'POST':
        if 'image' in request.form:
            # Process the captured frame from the camera
            image_data = request.form['image']
            image_data = image_data.split(',')[1]  # Remove the data URL prefix
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
            image = np.array(image)
        elif 'imagefile' in request.files:
            # Process the uploaded image
            imagefile = request.files['imagefile']
            image = np.asarray(Image.open(imagefile).convert('RGB'))
        else:
            return redirect(url_for('home'))

        waste_category, confidence, bin_image_path = process_image(image)

        recycling_data = get_recycling_data()  # Load the recycling data from the JSON file
        recycling_details = recycling_data.get(waste_category, {})
        details = recycling_details.get('details', '')
        website = recycling_details.get('website', '')

        return render_template('result.html', waste_category=waste_category, confidence=confidence, bin_image_path=bin_image_path, details=details, website=website)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
