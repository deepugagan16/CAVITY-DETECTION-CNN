from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('models/cavity_detection_model.h5')
img_width, img_height = 150, 150

# Function to process image and make predictions
def predict_cavity(image_path):
    image = Image.open(image_path)
    image = image.resize((img_width, img_height))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize the image
    prediction = model.predict(image)
    
    if prediction[0][0] > 0.5:
        return "The image does not contain a cavity."
    else:
        return "The image does contain a cavity."

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file uploaded"
        else:
            file = request.files['file']
            if file.filename == '':
                result = "No file selected"
            else:
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)
                result = predict_cavity(file_path)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

