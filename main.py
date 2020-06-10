import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img
from flask import Flask, render_template, request, redirect, flash, url_for
from keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename
import os


UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)

model = keras.models.load_model('Covid_Vgg.h5')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/index.html')
def returnhome():
    
    return render_template('index.html')    

@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    if request.method == 'POST':
        file=request.files['file']
        filename=secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #getPrediction(filename)
        print("Entered Prediction Logic")
        print("Loading model...")
        model = keras.models.load_model('Covid_Vgg.h5')
        print("Model loaded successfully")
        image = load_img(os.path.join(app.config['UPLOAD_FOLDER'],filename), target_size=(150, 150))
        print("Image is converted to 150*150")
        im_final = np.expand_dims(image, axis=0)
        print("Expanded dimension...")
        print("Started Prediction...")
        prediction = model.predict_classes(im_final)
        #print(prediction)
        for x in prediction:
            if x==0:
                result = "Patient has Covid +ve"
            else:
                result = "No Covid Symptoms Detected"
        flash(result)
    
    return render_template('about.html')
    

if __name__ == '__main__':
    app.run()
