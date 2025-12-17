import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load Keras model
model = tf.keras.models.load_model('./src/main/resources/Keras_Model.h5')

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((512, 512))
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Test images
images = [
    './src/main/resources/images/1.jpg',
    './src/main/resources/static/img/beagle.jpg',
    './src/main/resources/static/img/korgi.jpg'
]

classes = ['Beagle', 'Blenheim_spaniel', 'Dalmatian', 'German_shepherd', 'Husky', 'Labrador', 'Shar_pei', 'korgi', 'pug']

for img_path in images:
    if os.path.exists(img_path):
        img = preprocess_image(img_path)
        pred = model.predict(img)
        pred_class = np.argmax(pred, axis=1)[0]
        print(f"{os.path.basename(img_path)}: {classes[pred_class]} (index {pred_class})")
    else:
        print(f"{img_path} not found")