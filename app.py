import os

import numpy as np
from flask import Flask, request, render_template
from keras_preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename

model = load_model('my_model.h5')
# Classes of traffic signs
classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Vehicle > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing vehicle > 3.5 tons'}

app = Flask(__name__)
UPLOAD_FOLDER = "images/"
OUTPUT_FOLDER = "static/Done.jpg"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    global filepath, value
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if uploaded_file.filename != '':
        uploaded_file.save(UPLOAD_FOLDER + filename)
        filepath = os.path.realpath(UPLOAD_FOLDER + filename)

        img = image.load_img(filepath, target_size=(30, 30))  # load the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        values = np.argmax(model.predict(images, batch_size=32), axis=-1)  # predict the label for the image

        s = [str(i) for i in values]
        a = int("".join(s))
        value = "Predicted Traffic🚦Sign is: " + classes[a]

    return render_template('end.html', value_text=value)


if __name__ == "__main__":
    app.run(debug=True)
