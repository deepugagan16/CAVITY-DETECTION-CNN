<a id="readme-top"></a>

# Cavity Detection System

This project is a web-based AI system that detects dental cavities in images. The web interface allows users to upload images or take photos using their devices and then run predictions using a deep learning model.

## Features

- Upload images or take photos directly from the browser
- Predict whether the uploaded image contains a dental cavity
- Uses a Convolutional Neural Network (CNN) model built with Keras and TensorFlow
- Simple web interface built using Flask for quick interaction

## Installation

1. Clone the repository
``` sh
git clone https://github.com/deepugagan16/cavity-detection.git
cd cavity-detection
```

2. Create a virtual environment (optional but recommended)
``` sh
python -m venv venv
source venv/bin/activate
```

3. Install the dependencies
``` sh
pip install -r requirements.txt
```

4. Train the model (if not pre-trained)
Run the following command to train the model and save it:
``` sh
python train_model.py
```
This will generate the cavity_detection_model.h5 file in the models/ directory.

5. Run the Flask application
Make sure the trained model is in the models/ directory, then start the Flask application:
``` sh
python app.py
```
Visit http://127.0.0.1:5000/ in your browser to use the application.

## Usage

Scenario 1: Dental Clinics
Dentists can use this application to upload images of patients' oral cavities to quickly check for the presence of cavities. This tool can assist in diagnostic procedures and enhance patient care.

Scenario 2: Dental Training
Dental students and trainees can utilize this application to practice cavity detection and understand how different types of cavities appear in images. The tool provides a practical way to reinforce learning.

Scenario 3: Remote Diagnosis
Patients can upload images of their oral cavities from the comfort of their homes. The application will analyze the images and provide feedback, which can then be reviewed by a dental professional.

## Acknowledgments
* [Keras Documentation](https://keras.io/)
* [TensorFlow Documentation](https://www.tensorflow.org/guide)
* [Keras Documentation](https://flask.palletsprojects.com/en/latest/)

